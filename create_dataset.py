# %%
from datasets import load_dataset, Features, Value, ClassLabel, disable_caching
import pandas as pd

# %% load and save the entire raw dataset

# Disable caching temporarily
disable_caching()

ds = load_dataset("PatronusAI/HaluBench")

# # Define the features schema
# features = Features(
#     {
#         "id": Value("string"),
#         "passage": Value("string"),
#         "question": Value("string"),
#         "answer": Value("string"),
#         "label": ClassLabel(names=["FAIL", "PASS"]),
#         "source_ds": ClassLabel(
#             names=[
#                 "halueval",
#                 "covidQA",
#                 "pubmedQA",
#                 "DROP",
#                 "FinanceBench",
#                 "RAGTruth",
#             ]
#         ),
#     }
# )

# # Download the dataset with features
# ds = load_dataset("PatronusAI/HaluBench", features=features)

ds = load_dataset("PatronusAI/HaluBench")

# # Cast the features after loading
# ds = ds.cast(features)

# ds["test"].features

# Convert to pandas for easier manipulation
df = ds["test"].to_pandas()

df.to_csv("data/halubench/raw_halubench.csv", index=False)


# %%
def create_balanced_dataset(df, total_size, output_path):
    """
    We use RAGTruth, HaluEval, and PubMedQA, and FinanceBench
    Each kind of dataset takes equal proportion of total dataset
    Within each kind of dataset, 50% PASS and 50% FAIL

    takes in a dataframe, does the filtering, and saves it as a csv
    """
    # Define the datasets we want to use
    selected_datasets = ["RAGTruth", "halueval", "pubmedQA", "FinanceBench"]

    # Calculate samples needed per dataset
    samples_per_dataset = total_size // len(selected_datasets)
    # Ensure it's even for PASS/FAIL balance
    assert samples_per_dataset % 2 == 0

    final_dfs = []

    for dataset in selected_datasets:
        # Filter for current dataset
        dataset_df = df[df["source_ds"] == dataset]

        # Get balanced samples for PASS and FAIL
        samples_per_label = samples_per_dataset // 2

        # randomly sample some
        pass_samples = dataset_df[dataset_df["label"] == "PASS"].sample(
            n=samples_per_label, random_state=42
        )
        fail_samples = dataset_df[dataset_df["label"] == "FAIL"].sample(
            n=samples_per_label, random_state=42
        )

        # Combine PASS and FAIL samples
        balanced_dataset = pd.concat([pass_samples, fail_samples])
        final_dfs.append(balanced_dataset)

    # Combine all datasets
    final_df = pd.concat(final_dfs)

    # Save to CSV
    final_df.to_csv(output_path, index=False)

    print(f"Dataset creation complete. Total samples: {len(final_df)}")
    print("\nDistribution by source:")
    print(final_df["source_ds"].value_counts())
    print("\nDistribution by label:")
    print(final_df["label"].value_counts())

    return final_df


# %% 40 data points

df = pd.read_csv("data/halubench/raw_halubench.csv")

balanced_df = create_balanced_dataset(
    df, total_size=40, output_path="data/custom_40samples_unlabeled.csv"
)
