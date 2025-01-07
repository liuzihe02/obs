# %%
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from tqdm import tqdm
from typing import Optional

# Optional for UES/IDP, configure API key for desired model(s)
from dotenv import load_dotenv

load_dotenv()


def setup_models():
    # Initialize models (you'll need appropriate API keys set as env variables)
    models = {
        "gpt": ChatOpenAI(model="gpt-4o-mini"),
        # "claude": ChatAnthropic(model="claude-3-5-haiku-20241022"),
    }
    return models


def create_prompt(mode: str, fewshot_df: None | pd.DataFrame):
    # assert mode must be valid
    assert mode in {
        "base",
        "cot",
        "fewshot",
    }

    # if its fewshot, the fewshot dataframe must be provided
    if mode == "fewshot":
        assert fewshot_df is not None

    # Simple prompt template that includes context and asks for verification
    if mode == "base":
        template = """
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
        """

    elif mode == "cot":
        template = """In this task, you will be presented with a query, a reference text and an answer. The answer is
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

        EXPLANATION:"""

    elif mode == "fewshot":
        template = """In this task, you will be presented with a query, a reference text and an answer. The answer is 
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
        """

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


def evaluate_answers(
    eval_df: pd.DataFrame, mode: str, fewshot_df: None | pd.DataFrame = None
) -> pd.DataFrame:
    # Setup models and prompt
    models = setup_models()
    prompt = create_prompt(mode, fewshot_df)

    results = []

    # Evaluate each QA pair with each model
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df)):
        for model_name, model in models.items():
            # Format the prompt
            formatted_prompt = prompt.format(
                question=row["question"], passage=row["passage"], answer=row["answer"]
            )
            try:
                # Get model response
                response = model.invoke(formatted_prompt).content.strip()

                if mode == "base" or mode == "fewshot":
                    # get the response as an integer - and FLIP IT
                    # this is because we told the model 1 is hallucination, but for analysis 1 is faithful
                    print(formatted_prompt)
                    res = 1 - int(response)
                elif mode == "cot":
                    # the final integer in the string
                    res = 1 - int(response[-1])
            except ValueError:
                print(f"Error parsing response: {response}")

            # Store result
            results.append(
                {
                    "id": row["id"],
                    "eval_type": mode + "_" + model_name,
                    "eval_result": res,
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with original DataFrame
    merged_df = df.merge(results_df, on="id", how="left")
    return merged_df


# %% run the base hallucination detection
csv_path = "../data/custom_16samples_fewshot.csv"
df = pd.read_csv(csv_path)
results = evaluate_answers(eval_df=df, mode="base")

# Save results
results.to_csv("../data/custom_16samples_eval_llm-judge-base.csv", index=False)

# %% fewshot
csv_path = "../data/custom_16samples_fewshot.csv"
df = pd.read_csv(csv_path)
results = evaluate_answers(eval_df=df, mode="fewshot", fewshot_df=df)

# Save results
results.to_csv("../data/custom_16samples_eval_llm-judge-fewshot.csv", index=False)
