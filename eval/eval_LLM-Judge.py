# %%
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm_asyncio
from typing import Literal, Optional, Union

# Optional for UES/IDP, configure API key for desired model(s)
from dotenv import load_dotenv

load_dotenv()

# specify prompt templates
PROMPT_TEMPLATES = {
    # Simple prompt template that includes context and asks for verification
    "base": """
        In this task, you will be presented with a query, a reference text and an answer. The answer is
        generated to the question based on the reference text. The answer may contain false information. Your objective is to determine whether the answer text contains factual information and is not a hallucination. A 'hallucination' refers to
        an answer that is not based on the reference text or assumes information that is not available in
        the reference text. Please read the query and reference text carefully before determining
        your response. Your response should be a SINGLE number: either 1 or 0, and it should NOT include any other text or characters like ". 1 indicates hallucinations and 0 indicates no hallucinations (faithful to reference text)

            [BEGIN DATA]
            ************
            [Query]: {question}
            ************
            [Reference text]: {passage}
            ************
            [Answer]: {answer}
            ************
            [END DATA]

            Does the answer contain hallucinations?
        """,
    "cot": """In this task, you will be presented with a query, a reference text and an answer. The answer is
        generated to the question based on the reference text. The answer may contain false information. Your objective is to determine whether the answer text contains factual information and is not a hallucination. A 'hallucination' refers to
        an answer that is not based on the reference text or assumes information that is not available in
        the reference text. 

        [BEGIN DATA]
        ************
        [Query]: {question}
        ************
        [Reference text]: {passage}
        ************
        [Answer]: {answer}
        ************
        [END DATA]

        Does the answer contain hallucinations?

        Please read the query, reference text and answer carefully, then write out in a step by step manner
        an EXPLANATION to determine if a 'hallucination' is present. Avoid simply
        stating the correct answer at the outset. END your response with LABEL, which should be a a SINGLE number: either 1 or 0, and it should NOT include any other text or characters like ". 1 indicates hallucinations and 0 indicates no hallucinations (faithful to reference text).

        Example response:
        ************
        EXPLANATION: An explanation of your reasoning for why 'hallucination' is present
        LABEL: 1 or 0
        ************

        EXPLANATION:""",
    "fewshot": """In this task, you will be presented with a query, a reference text and an answer. The answer is 
        generated to the question based on the reference text. The answer may contain false information. Your objective is to determine whether the answer text contains factual information and is not a hallucination. A 'hallucination' refers to 
        an answer that is not based on the reference text or assumes information that is not available in the reference text.

        Here are some examples to help you understand the task:

        {examples}

        Now, please evaluate the following case:

        [BEGIN DATA]
        ************
        [Query]: {question}
        ************
        [Reference text]: {passage}
        ************
        [Answer]: {answer}
        ************
        [END DATA]

        Does the answer contain hallucinations? Please respond with ONLY a single number: 1 for hallucination or 0 for no hallucination (faithful to reference text), and it should NOT include any other text or characters like "
        """,
}


def setup_models():
    # Initialize models (you'll need appropriate API keys set as env variables)
    models = {
        "gpt": ChatOpenAI(model="gpt-4o-mini"),
        # "claude": ChatAnthropic(model="claude-3-5-haiku-20241022"),
    }
    return models


def create_prompt(
    mode: Literal["base", "cot", "fewshot"], fewshot_df: None | pd.DataFrame
) -> PromptTemplate:
    template = PROMPT_TEMPLATES[mode]

    # if its fewshot, the fewshot dataframe must be provided
    if mode == "fewshot":
        assert fewshot_df is not None

        # Generate examples string from fewshot DataFrame
        examples_text = ""
        for _, row in fewshot_df.iterrows():
            examples_text += f"""
            Example:
            ************
            [Query]: {row['question']}
            ************
            [Reference text]: {row['passage']}
            ************
            [Answer]: {row['answer']}
            ************
            Label: {0 if row['label']=="PASS" else 1}
            """
            # remember to flip the label according to our convention

        # Replace the {examples} placeholder with actual examples
        template = template.format(
            examples=examples_text,
            question="{question}",
            passage="{passage}",
            answer="{answer}",
        )
    return PromptTemplate.from_template(template)


# asynchronous calls
async def evaluate_single_qa(
    row: pd.Series,
    model_name: str,
    model: BaseLanguageModel,
    prompt_template: PromptTemplate,
    mode: Literal["base", "cot", "fewshot"],
) -> Optional[dict[str, Union[str, int]]]:
    # Format the prompt
    formatted_prompt = prompt_template.format(
        question=row["question"], passage=row["passage"], answer=row["answer"]
    )

    print(formatted_prompt)

    try:
        # Get model response, this part can let other tasks use
        # use ainvoke for async invoke
        response = await model.ainvoke(formatted_prompt)
        # some formatting
        message = response.content.strip()

        if mode == "base" or mode == "fewshot":
            # get the response as an integer - and FLIP IT
            # this is because we told the model 1 is hallucination, but for analysis 1 is faithful
            res = 1 - int(message)
        elif mode == "cot":
            # the final integer in the string
            res = 1 - int(message[-1])

        # Store result
        return {
            "id": row["id"],
            "eval_type": mode + "_" + model_name,
            "eval_result": res,
        }
    except ValueError:
        print(f"Error parsing response: {message}")
        return None


async def evaluate_answers(
    eval_df: pd.DataFrame,
    mode: Literal["base", "cot", "fewshot"],
    fewshot_df: None | pd.DataFrame = None,
) -> pd.DataFrame:
    # Setup models and prompt
    models = setup_models()
    prompt_template = create_prompt(mode, fewshot_df)

    tasks = []
    for _, row in eval_df.iterrows():
        for model_name, model in models.items():
            # define the function
            task = evaluate_single_qa(row, model_name, model, prompt_template, mode)
            # add this task
            tasks.append(asyncio.create_task(task))

    # Use asyncio.gather to run tasks concurrently
    results = []
    for result in tqdm(await asyncio.gather(*tasks)):
        if result:
            results.append(result)

    # convert to dataframe
    results_df = pd.DataFrame([r for r in results if r is not None])

    # Merge results with original DataFrame
    merged_df = eval_df.merge(results_df, on="id", how="left", validate="1:1")
    return merged_df


# %% Modified run section
async def run_evaluation():
    csv_path = "../data/custom_16samples_fewshot.csv"
    df = pd.read_csv(csv_path)

    # # Run base evaluation
    # base_results = await evaluate_answers(eval_df=df, mode="base")
    # base_results.to_csv("../data/custom_16samples_eval_llm-judge-base.csv", index=False)

    # Run fewshot evaluation
    fewshot_results = await evaluate_answers(eval_df=df, mode="fewshot", fewshot_df=df)
    fewshot_results.to_csv(
        "../data/custom_16samples_eval_llm-judge-fewshot.csv", index=False
    )


# For Jupyter notebook, use this:
await run_evaluation()

# For regular Python script, use this instead:
# if __name__ == "__main__":
#     asyncio.run(run_evaluation())

# %%
