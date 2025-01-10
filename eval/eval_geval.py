# %%
import deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, BaseMetric
from deepeval.dataset import EvaluationDataset
import pytest
import pandas as pd
from typing import List
import os

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# %% preprocess data
# TODO: KEEP RUNNING THESE FEW CELLS UNTIL WE ARE DONE WITH EVALS

# Initialize results file if it doesn't exist
results_path = "../data/eval_geval_custom_1000samples.csv"
if not os.path.exists(results_path):
    empty_df = pd.DataFrame(columns=["id", "eval_type", "eval_result", "eval_reason"])
    empty_df.to_csv(results_path, index=False)
    print("Created new results file with headers")


def filter_batch(eval_df: pd.DataFrame, results_df: pd.DataFrame, batch_size: int):
    """since geval doesnt work for large batch size, we do batching ourselves
    this function will filter batch_size number of datapoints we have NOT evaluated yet"""

    # Try to load existing results
    done_ids = set(results_df["id"])
    print(f"Found {len(done_ids)} completed evaluations")

    # Get remaining IDs
    remaining_ids = set(eval_df["id"]) - done_ids
    if not remaining_ids:
        print("===DONE WITH EVALS===")
        return

    # Select batch_size number of remaining IDs
    batch_ids = list(remaining_ids)[:batch_size]

    # Filter eval_df to only include the selected batch_ids
    batch_df = eval_df[eval_df["id"].isin(batch_ids)].copy()

    print(
        f"Processing batch of {len(batch_ids)} items. {len(remaining_ids) - len(batch_ids)} items remaining."
    )
    return batch_df


# load from csv into ragas dataset for evaluation
def process_data_deepeval(dataset_df: pd.DataFrame) -> EvaluationDataset:
    """turn a dataframe into a geval dataset"""
    # Initialize list to store test cases
    test_cases: List[LLMTestCase] = []

    # Iterate through DataFrame rows
    for idx, row in dataset_df.iterrows():
        # Create test case for each row
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            context=[row["passage"]],  # If you need context, must be a list of strings
        )
        test_cases.append(test_case)

    # Create deepeval EvaluationDataset from test cases
    eval_dataset = EvaluationDataset(test_cases=test_cases)
    return eval_dataset


# Load the dataset from the TSV file
eval_df = pd.read_csv("../data/custom_1000samples.csv")
results_df = pd.read_csv("../data/eval_geval_custom_1000samples.csv")
# filter only those we have not done yet
# USE A SMALLER BATCH SIZE
eval_df = filter_batch(eval_df, results_df, 100)
eval_dataset = process_data_deepeval(eval_df)


# %% define metric
halu_metric = GEval(
    name="Hallucinations",
    criteria="""You will be presented with a query (INPUT), a reference text (CONTEXT) and an answer(ACTUAL_OUTPUT). The answer is
        generated to the question based on the reference text. The answer may contain false information. Your objective is to determine whether the answer text contains factual information and is not a hallucination. A 'hallucination' refers to an answer that is not based on the reference text or assumes information that is not available in
        the reference text. Please read the query and reference text carefully before determining your response. 1 indicates strong hallucinations and 0 indicates no hallucinations (faithful to reference text)""",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
    model="gpt-4o",
    # binary metric
    strict_mode=True,
    async_mode=True,
    # view the COT
    verbose_mode=False,
)


# %%

# After running evaluation
eval_result = eval_dataset.evaluate(metrics=[halu_metric])

# Create list to store results
all_results = []

# Create a dictionary to mafind the relevant id in eval_df
answer_to_id = dict(zip(eval_df["answer"], eval_df["id"]))

# Extract scores and reasons from each test result
for test_result in eval_result.test_results:
    result_dict = {
        "id": answer_to_id[test_result.actual_output],
        "eval_type": "geval_hallucination",
        "eval_result": int(test_result.metrics_data[0].score),
        "eval_reason": test_result.metrics_data[0].reason,
    }

    all_results.append(result_dict)

results_df = pd.DataFrame(all_results)

# use append mode
results_df.to_csv(results_path, mode="a", header=False, index=False)


# %%
