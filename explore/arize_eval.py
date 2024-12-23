# %% Load in imports
import pandas as pd
from datasets import load_dataset
import sys
from pathlib import Path

# arize imports
import nest_asyncio
from phoenix.evals import HallucinationEvaluator, OpenAIModel, QAEvaluator, run_evals

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR

# load in environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# %% prepare dataset

qa_data = load_dataset("json", data_files=str(DATA_DIR / "qa_data.json"))

# Convert the huggingface data to pandas
qa_df = qa_data["train"].to_pandas()

# at this point, this is in "knowledge, question, right_answer, hallucinated answer"
# we will take the first n rows, then split into 2, with the first half being right_answer and second half being hallucinated

qa_df = qa_df.head(10)

# Create two separate dataframes
df_right = pd.DataFrame(
    {
        "reference": qa_df["knowledge"],
        "query": qa_df["question"],
        "response": qa_df["right_answer"],
    }
)

df_hallucinated = pd.DataFrame(
    {
        "reference": qa_df["knowledge"],
        "query": qa_df["question"],
        "response": qa_df["hallucinated_answer"],
    }
)

# Concatenate vertically (stack on top of each other)
# top half is right, bottom half is hallucinated
qa_df = pd.concat([df_right, df_hallucinated], axis=0, ignore_index=True)

# %% Evaluate and Log Results

nest_asyncio.apply()  # This is needed for concurrency in notebook environments

# change the model ehre
eval_model = OpenAIModel(model="gpt-4o-mini")

# Define your evaluators
hallucination_evaluator = HallucinationEvaluator(eval_model)
qa_evaluator = QAEvaluator(eval_model)

# We have to make some minor changes to our dataframe to use the column names expected by our evaluators
# for `hallucination_evaluator` the input df needs to have columns 'output', 'input', 'context'
# for `qa_evaluator` the input df needs to have columns 'output', 'input', 'reference'
qa_df["context"] = qa_df["reference"]
qa_df.rename(columns={"query": "input", "response": "output"}, inplace=True)
assert all(
    column in qa_df.columns for column in ["output", "input", "context", "reference"]
)

# Run the evaluators, each evaluator will return a dataframe with evaluation results
# We upload the evaluation results to Phoenix in the next step

# uses simple prompt attached to the query/reference/answer
hallucination_eval_df, qa_eval_df = run_evals(
    dataframe=qa_df,
    evaluators=[hallucination_evaluator, qa_evaluator],
    provide_explanation=True,
)

# %% Analyze results
results_df = qa_df.copy()
results_df["hallucination_eval"] = hallucination_eval_df["label"]
results_df["hallucination_explanation"] = hallucination_eval_df["explanation"]
results_df["qa_eval"] = qa_eval_df["label"]
results_df["qa_explanation"] = qa_eval_df["explanation"]
results_df.head()

# %%manual exploration
print(results_df["output"][10])

# %%
