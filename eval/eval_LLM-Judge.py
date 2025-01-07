# %%
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
import os
import pandas as pd
from tqdm import tqdm

# Optional for UES/IDP, configure API key for desired model(s)
from dotenv import load_dotenv

load_dotenv()


def setup_models():
    # Initialize models (you'll need appropriate API keys set as env variables)
    models = {
        "gpt-4": ChatOpenAI(model="gpt-4o-mini"),
        "claude": ChatAnthropic(model="claude-3-5-haiku-20241022"),
    }
    return models


def create_base_prompt():
    # Simple prompt template that includes context and asks for verification
    template = """
    In this task, you will be presented with a query, a reference text and an answer. The answer is
    generated to the question based on the reference text. The answer may contain false information. Your objective is to determine whether the answer text contains factual information and is not a hallucination. A 'hallucination' refers to
    an answer that is not based on the reference text or assumes information that is not available in
    the reference text. Please read the query and reference text carefully before determining
    your response. Your response should be a single number: either "1" or "0", and it should not include any other text or characters. "1" indicates hallucinations and "0" indicates no hallucinations (faithful to reference text)

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
    return PromptTemplate.from_template(template)


def evaluate_answers(df: pd.DataFrame) -> pd.DataFrame:
    # Setup models and prompt
    models = setup_models()
    prompt = create_base_prompt()

    results = []

    # Evaluate each QA pair with each model
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        for model_name, model in models.items():
            # Format the prompt
            formatted_prompt = prompt.format(
                question=row["question"], passage=row["passage"], answer=row["answer"]
            )

            # Get model response
            response = model.invoke(formatted_prompt).content.strip()

            # get the response as an integer - and FLIP IT
            # this is because we told the model 1 is hallucination, but for analysis 1 is faithful
            res = 1 - int(response)

            # Store result
            results.append({"id": row["id"], "model": model_name, "eval": res})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Merge results with original DataFrame
    merged_df = df.merge(results_df, on="id", how="left")
    return merged_df


# %%


csv_path = "../data/custom_16samples_fewshot.csv"
df = pd.read_csv(csv_path)
results = evaluate_answers(df)

# %%

# Save results
results.to_csv("../data/custom_16samples_eval_llm-judge.csv", index=False)
