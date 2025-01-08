# %%
from datasets import load_dataset, Features, Value, ClassLabel, disable_caching
import pandas as pd

# %% load and save the entire raw dataset

# Disable caching temporarily
disable_caching()

ds = load_dataset("PatronusAI/HaluBench")

# # Define the features schema
# features = Features(
#
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
            n=samples_per_label, random_state=41
        )
        fail_samples = dataset_df[dataset_df["label"] == "FAIL"].sample(
            n=samples_per_label, random_state=41
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


df = pd.read_csv("data/halubench/raw_halubench.csv")

# %% 40 data points

full_df = create_balanced_dataset(
    df, total_size=40, output_path="data/custom_40samples_full.csv"
)

# %% create 10 data points for few shot prompting
fewshot_df = create_balanced_dataset(
    df, total_size=16, output_path="data/custom_16samples_fewshot.csv"
)

# %%
# load back in the 2 datasets
# make sure the full_df and fewshot_df dont have overlapping examples
# Get sets of IDs from both dataframes
# Load the saved datasets
full_df = pd.read_csv("data/custom_40samples_full.csv")
fewshot_df = pd.read_csv("data/custom_16samples_fewshot.csv")

full_df_ids = set(full_df["id"])
fewshot_df_ids = set(fewshot_df["id"])

# Find any overlapping IDs
overlapping_ids = full_df_ids.intersection(fewshot_df_ids)

# Check and print results
if len(overlapping_ids) > 0:
    print(f"WARNING: Found {len(overlapping_ids)} overlapping examples!")
    print("Overlapping IDs:", overlapping_ids)
    raise AssertionError("Datasets should not have overlapping examples")
else:
    print("Success: No overlapping examples found between the datasets")
    print(f"Number of unique IDs in full dataset: {len(full_df_ids)}")
    print(f"Number of unique IDs in few-shot dataset: {len(fewshot_df_ids)}")


# %% filter datasets for pubmedqa and ragtruth only, because RAGAS only accepts long form context
full_df = pd.read_csv("data/custom_40samples_full.csv")
ragas_df = full_df[full_df["source_ds"].isin(["pubmedQA", "RAGTruth"])]

# Save filtered dataset
ragas_df.to_csv("data/custom_40samples_full_ragas.csv", index=False)

# %%
