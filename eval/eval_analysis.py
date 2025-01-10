# %%
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
import os


def evaluate_predictions(y_true, y_pred, eval_type):
    """Calculate and print evaluation metrics for predictions"""

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("=" * 20 + "START" + "=" * 20)
    print(f"\nResults for {eval_type}:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("=" * 20 + "END" + "=" * 20)
    print("\n")


def analyze_eval_files(folder_path: str, dataset: str):
    """Analyze all evaluation files in the given folder. Allows analysis to be done on the subset of the dataset, or the entire dataset"""

    # Assert valid dataset
    assert dataset in ["all", "RAGTruth", "halueval", "pubmedQA", "FinanceBench"]

    # Read ground truth file
    gt_df = pd.read_csv(os.path.join(folder_path, "custom_1000samples.csv"))

    # Check whether all the datapoint ids are unique
    print(f"Total rows: {len(gt_df)}")
    print(f"Unique IDs: {len(gt_df['id'].unique())}")
    if len(gt_df) != len(gt_df["id"].unique()):
        print(
            "Duplicate IDs found:",
            gt_df["id"].value_counts()[gt_df["id"].value_counts() > 1],
        )

    # Filter ground truth for specific dataset if not "all"
    if dataset != "all":
        gt_df = gt_df[gt_df["source_ds"] == dataset]

    if gt_df.empty:
        raise ValueError(f"No samples found for dataset {dataset}")

    # Find all eval files
    eval_files = [
        f
        for f in os.listdir(folder_path)
        if f.startswith("eval") and f.endswith(".csv")
    ]

    # Process each eval file
    for eval_file in eval_files:
        # Read evaluation file
        cur_df = pd.read_csv(os.path.join(folder_path, eval_file))
        # make ids explicitly string
        cur_df["id"] = cur_df["id"].astype(str)

        # Get unique eval types in this file
        eval_types = cur_df["eval_type"].unique()

        # Process each eval type
        for eval_type in eval_types:
            # Filter data for current eval type
            eval_type_df: pd.DataFrame = cur_df[cur_df["eval_type"] == eval_type]

            # Merge with ground truth based on id. unner means we only keep rows where ids exist in BOTH dataframes!
            merged_df = eval_type_df.merge(gt_df[["id", "label"]], on="id", how="inner")

            if not merged_df.empty:
                # Convert to binary format
                y_true = (merged_df["label"] == "PASS").astype(int)
                y_pred = merged_df["eval_result"]

                # Calculate and print metrics
                evaluate_predictions(y_true, y_pred, f"{eval_file} - {eval_type}")
            else:
                print(f"\nWARNING: No matching IDs found for {eval_file} - {eval_type}")


# %% run all analysis
folder_path = "../data"  # Adjust this to your data folder path
analyze_eval_files(folder_path, "FinanceBench")

# %%
