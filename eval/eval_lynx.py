# %%
import pandas as pd
from patronus import Client, task, TaskResult, Dataset
import os

# Initialize Patronus client
cli = Client(api_key=os.getenv("PATRONUS_API_KEY"))
fi = cli.remote_dataset("financebench").dataset

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
eval_df = pd.read_csv("../data/custom_16samples.csv")
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

# Create a dictionary to mafind the relevant id in eval_df
answer_to_id = dict(zip(eval_df["answer"], eval_df["id"]))

# Process results into same format as original
processed_results = []
for _, row in df_results.iterrows():
    result_dict = {
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
results_df.to_csv("../data/eval_lynx_custom_16samples.csv", index=False)

# Print summary statistics
print("\nEvaluation Summary:")
print(f"Total samples evaluated: {len(results_df)}")
print(f"Hallucination rate: {(1 - results_df['eval_result'].mean()) * 100:.2f}%")

# %%
