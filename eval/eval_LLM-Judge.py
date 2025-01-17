# %%
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.base import BaseLanguageModel
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from tqdm import tqdm
import asyncio
from asyncio import Semaphore
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
        an EXPLANATION to determine if a 'hallucination' is present. Avoid simply stating the correct answer at the outset. END the very last token of your response with LABEL, which should be a a SINGLE number: either 1 or 0, and it should NOT include any other text or characters like ". 1 indicates hallucinations and 0 indicates no hallucinations (faithful to reference text).

        Example response:

        EXPLANATION: An explanation of your reasoning for why 'hallucination' is present
        LABEL: 1 or 0

        EXPLANATION:""",
    "fewshot": """In this task, you will be presented with a query, a reference text and an answer. The answer is 
        generated to the question based on the reference text. The answer may contain false information. Your objective is to determine whether the answer text contains factual information and is not a hallucination. A 'hallucination' refers to an answer that is not based on the reference text or assumes information that is not available in the reference text.

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
        # "gpt-4o": ChatOpenAI(model="gpt-4o"),
        # "gpt-4o-mini": ChatOpenAI(model="gpt-4o-mini"),
        # "3.5haiku": ChatAnthropic(model="claude-3-5-haiku-latest"),
        "3.5sonnet": ChatAnthropic(model="claude-3-5-sonnet-latest"),
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


async def process_batch(
    row: pd.Series,
    prompt_template: PromptTemplate,
    model_name: str,
    model: BaseLanguageModel,
    mode: Literal["base", "cot", "fewshot"],
    num_samples: int = 1,
) -> Optional[dict[str, Union[str, int]]]:
    """Makes multiple of the SAME API call and process responses
    then averages the results"""

    formatted_prompt = prompt_template.format(
        question=row["question"], passage=row["passage"], answer=row["answer"]
    )

    # stores the results of multiple of the same calls
    results = []
    for _ in range(num_samples):
        try:
            response = await model.ainvoke(formatted_prompt)
            message = response.content.strip()

            if mode == "base" or mode == "fewshot":
                res = 1 - int(message)
            elif mode == "cot":
                res = 1 - int(message[-1])
                # note that this task will not proceed sequentially!
            results.append(res)

        except ValueError as e:
            print(f"Error processing response: {e}")
            print("prompt was", formatted_prompt)
            print("message was", message[-100:])
            return None

    # check concurrency
    # print(formatted_prompt[700:1200], "======", results)

    return {
        "id": row["id"],
        "eval_type": mode + "_" + model_name + "_numsamples" + str(num_samples),
        "eval_result": round(sum(results) / len(results)),
        # round basically converts a float to 0/1 binary labels
    }


async def evaluate(
    eval_df: pd.DataFrame,
    mode: Literal["base", "cot", "fewshot"],
    num_samples: int = 1,
    max_crr: int = 20,
    fewshot_df: None | pd.DataFrame = None,
) -> pd.DataFrame:
    # num samples is for chainpoll or basepoll

    # limits how many api calls we can make at once
    semaphore = Semaphore(max_crr)

    async def limited_process_batch(
        row, prompt_template, model_name, model, mode, num_samples
    ):
        async with semaphore:  # Limit concurrent process_batch tasks
            return await process_batch(
                row, prompt_template, model_name, model, mode, num_samples
            )

    # Setup models and prompt
    models = setup_models()
    prompt_template = create_prompt(mode, fewshot_df)

    tasks = []
    for _, row in eval_df.iterrows():
        for model_name, model in models.items():
            task = limited_process_batch(
                row, prompt_template, model_name, model, mode, num_samples
            )
            tasks.append(asyncio.create_task(task))

    results = []
    for result in await tqdm_asyncio.gather(*tasks, desc=f"Running {mode} evaluation"):
        if result:
            results.append(result)

    results_df = pd.DataFrame(results)
    return results_df


# %% Modified run section
async def run_evaluation():
    full_df = pd.read_csv("../data/custom_1000samples.csv")
    fewshot_df = pd.read_csv("../data/custom_16samples.csv")

    # # Run base evaluation
    # base_results = await evaluate(
    #     eval_df=full_df, mode="base", num_samples=1, max_crr=2
    # )
    # base_results.to_csv(
    #     "../data/eval_llm-judge-base_custom_1000samples.csv",
    #     index=False,
    #     header=False,
    #     mode="a",
    # )  # append mode

    # # basepoll
    # base_results = await evaluate(
    #     eval_df=full_df, mode="base", num_samples=5, max_crr=100
    # )
    # base_results.to_csv(
    #     "../data/eval_llm-judge-basepoll_custom_16samples.csv", index=False
    # )

    # # Run fewshot evaluation
    # fewshot_results = await evaluate(
    #     eval_df=full_df, mode="fewshot", fewshot_df=fewshot_df, max_crr=100
    # )
    # fewshot_results.to_csv(
    #     "../data/eval_llm-judge-fewshot_custom_1000samples.csv", index=False
    # )

    # cot
    base_results = await evaluate(eval_df=full_df, mode="cot", num_samples=1, max_crr=3)
    base_results.to_csv(
        "../data/eval_llm-judge-cot_custom_1000samples.csv",
        index=False,
        header=False,
        mode="a",
    )  # append mode

    # # chainpoll
    # base_results = await evaluate(
    #     eval_df=full_df, mode="cot", num_samples=5, max_crr=150
    # )
    # base_results.to_csv(
    #     "../data/eval_llm-judge-chainpoll_custom_1000samples.csv",
    #     index=False,
    #     header=False,
    #     mode="a",
    # )


# For Jupyter notebook, use this:
await run_evaluation()

# For regular Python script, use this instead:
# if __name__ == "__main__":
#     asyncio.run(run_evaluation())

# %%
