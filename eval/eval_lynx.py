# %%
import pandas as pd
from patronus import Client, task, TaskResult, Dataset
import os

# Initialize Patronus client
cli = Client(api_key=os.getenv("PATRONUS_API_KEY"))


# %%
# Load and prepare the dataset
def process_data_lynx(df: pd.DataFrame) -> Dataset:
    # Define column mapping
    column_map = {
        "question": "evaluated_model_input",
        "answer": "evaluated_model_output",
        "passage": "evaluated_model_retrieved_context",
    }

    # Rename columns and keep only the ones we need
    df = df[column_map.keys()].rename(columns=column_map)

    return Dataset.from_dataframe(df)


# Load dataset
eval_df = pd.read_csv("../data/custom_1000samples.csv")

# Add this check after loading eval_df
duplicate_ids = eval_df["id"].duplicated()
if duplicate_ids.any():
    print("Found duplicate IDs in input data:")
    print(eval_df[duplicate_ids]["id"].tolist())

# Create a dictionary to mafind the relevant id in eval_df
answer_to_id = dict(zip(eval_df["answer"], eval_df["id"]))

# filter only the ones we havent evalauted yet
# Load existing evaluation results if file exists
results_path = "../data/eval_lynx_custom_1000samples.csv"
if os.path.exists(results_path):
    existing_results = pd.read_csv(results_path)
    # Filter out already evaluated IDs
    eval_df = eval_df[~eval_df["id"].isin(existing_results["id"])]
    print("Im about to evaluate", eval_df["id"])

# If there are no new samples to evaluate, exit
if len(eval_df) == 0:
    print("All samples have been evaluated already.")
    exit()
else:
    print("evaluating " + str(len(eval_df)) + " samples")

eval_dataset = process_data_lynx(eval_df)

# Define evaluator - using Lynx-large for hallucination detection
lynx_hallucination = cli.remote_evaluator(
    evaluator_id_or_alias="lynx-small", criteria="patronus:hallucination"
)

# %%
# Run experiment. :NOTE:please run this using jupyter cells
results = await cli.experiment(
    project_name="halu",
    data=eval_dataset,
    evaluators=[lynx_hallucination],
)

# %%

# Convert results to DataFrame
df_results = results.to_dataframe()

# %%


# Process results into same format as original
processed_results = []
for _, row in df_results.iterrows():
    result_dict = {
        # this causes some problems for finance bench, as the outputs are duplicates!!
        "id": answer_to_id[row["evaluated_model_output"]],
        "eval_type": "lynx",
        "eval_result": int(
            row["pass"]
        ),  # Convert boolean to int. 1 is faithful no halu
        "eval_reason": row["evaluation_explanation"],
    }
    processed_results.append(result_dict)

# Save results
results_df = pd.DataFrame(processed_results)
# use append mode
results_df.to_csv(results_path, index=False, header=False, mode="a")

# Print summary statistics
print("\nEvaluation Summary:")
print(f"Total samples evaluated: {len(results_df)}")
print(f"Hallucination rate: {(1 - results_df['eval_result'].mean()) * 100:.2f}%")

# %% do some post processing

# Load the results file
results_path = "../data/eval_lynx_custom_1000samples.csv"
df = pd.read_csv(results_path)

# Find and print duplicates
duplicate_mask = df.duplicated(subset=["id"], keep=False)
if duplicate_mask.any():
    print("\nFound duplicate IDs:")
    duplicates = df[duplicate_mask].sort_values("id")
    print(duplicates[["id", "eval_result", "eval_reason"]])
    print(
        f"\nNumber of duplicate rows: {len(duplicates) - len(duplicates['id'].unique())}"
    )

# Remove duplicate rows while keeping the first occurrence
df_cleaned = df.drop_duplicates(subset=["id"], keep="first")

# Print summary statistics
print("\nCleaning Summary:")
print(f"Original number of rows: {len(df)}")
print(f"Rows after cleaning: {len(df_cleaned)}")
print(f"Number of rows removed: {len(df) - len(df_cleaned)}")

# Don't save the changes, just show what would be removed

# %%
# Save the cleaned dataset
df_cleaned.to_csv(results_path, index=False)
print(f"\nCleaned dataset saved to {results_path}")

# %%
