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
def create_balanced_dataset(df: pd.DataFrame, full_size: int, fewshot_size: int):
    """
    We use RAGTruth, HaluEval, and PubMedQA, and FinanceBench
    Each kind of dataset takes equal proportion of total dataset
    Within each kind of dataset, 50% PASS and 50% FAIL

    takes in a dataframe, does the filtering, and saves it as a csv
    """
    # Define the datasets we want to use
    selected_datasets = ["RAGTruth", "halueval", "pubmedQA", "FinanceBench"]

    # Calculate samples needed per dataset
    full_samples_per_dataset = full_size // len(selected_datasets)
    fewshot_samples_per_dataset = fewshot_size // len(selected_datasets)
    # Ensure they're even for PASS/FAIL balance
    assert full_samples_per_dataset % 2 == 0, "full_size must be divisible by 8"
    assert fewshot_samples_per_dataset % 2 == 0, "fewshot_size must be divisible by 8"

    full_dfs = []
    fewshot_dfs = []

    for dataset in selected_datasets:
        # Filter for current dataset
        dataset_df = df[df["source_ds"] == dataset]

        # Calculate samples needed per label
        full_samples_per_label = full_samples_per_dataset // 2
        fewshot_samples_per_label = fewshot_samples_per_dataset // 2

        for label in ["PASS", "FAIL"]:
            # Get all samples for this dataset and label
            label_df: pd.DataFrame = dataset_df[dataset_df["label"] == label]

            # Randomly sample without replacement for both datasets
            total_needed = full_samples_per_label + fewshot_samples_per_label
            all_samples = label_df.sample(n=total_needed, random_state=42)

            # Split into full and few-shot
            full_samples = all_samples.iloc[:full_samples_per_label]
            fewshot_samples = all_samples.iloc[full_samples_per_label:]

            full_dfs.append(full_samples)
            fewshot_dfs.append(fewshot_samples)

        # Combine all samples
    full_df = pd.concat(full_dfs)
    fewshot_df = pd.concat(fewshot_dfs)

    # Print statistics
    print(f"Dataset creation complete.")
    print(f"\nFull dataset ({len(full_df)} samples) distribution by source:")
    print(full_df["source_ds"].value_counts())
    print("\nDistribution by label:")
    print(full_df["label"].value_counts())

    print(f"\nFew-shot dataset ({len(fewshot_df)} samples) distribution by source:")
    print(fewshot_df["source_ds"].value_counts())
    print("\nDistribution by label:")
    print(fewshot_df["label"].value_counts())

    return full_df, fewshot_df


df = pd.read_csv("data/halubench/raw_halubench.csv")

# %% bigger dataset

full_df, fewshot_df = create_balanced_dataset(df, 1000, 16)
full_df.to_csv("data/custom_1000samples.csv")
fewshot_df.to_csv("data/custom_16samples.csv")

# %%
# load back in the 2 datasets
# make sure the full_df and fewshot_df dont have overlapping examples
# Get sets of IDs from both dataframes
# Load the saved datasets
full_df = pd.read_csv("data/custom_1000samples.csv")
fewshot_df = pd.read_csv("data/custom_16samples.csv")

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
full_df = pd.read_csv("data/custom_16samples.csv")
ragas_df = full_df[full_df["source_ds"].isin(["pubmedQA", "RAGTruth"])]

# Save filtered dataset
ragas_df.to_csv("data/custom_16samples_ragas.csv", index=False)

# %%
