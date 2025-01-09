# %%
import deepeval

from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric, BaseMetric
from deepeval.dataset import EvaluationDataset
import pytest
import pandas as pd
from typing import List

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

# %% preprocess data


# load from csv into ragas dataset for evaluation
def process_data_deepeval(dataset_df: pd.DataFrame) -> EvaluationDataset:
    # Initialize list to store test cases
    test_cases: List[LLMTestCase] = []

    # Iterate through DataFrame rows
    for idx, row in dataset_df.iterrows():
        # Create test case for each row
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            context=[row["passage"]],  # If you need context, must be a list of strings
            additional_metadata={"id": row["id"]},
        )
        test_cases.append(test_case)

    # Create deepeval EvaluationDataset from test cases
    eval_dataset = EvaluationDataset(test_cases=test_cases)
    return eval_dataset


# Load the dataset from the TSV file
eval_df = pd.read_csv("../data/custom_16samples_fewshot.csv")
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
    model="gpt-4o-mini",
    # binary metric
    strict_mode=True,
    async_mode=True,
    # view the COT
    verbose_mode=False,
)

# %% run the evals

# """ to run several tests in parallel, do `deepeval test run test_bulk.py -n 3` where -n determines the number of processes to use"""

# results = []


# @pytest.mark.parametrize(
#     "test_case",
#     eval_dataset,
# )
# def test_all(test_case: LLMTestCase):
#     # assert_test(test_case, [halu_metric])
#     halu_metric.measure(test_case)
#     results.append((halu_metric.score, halu_metric.reason))


# @deepeval.on_test_run_end
# def function_to_be_called_after_test_run():
#     print("Test finished! Printing results")
#     print(results)


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

results_df.to_csv("../data/eval_geval_custom_16samples.csv")


# %%
