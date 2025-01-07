# %%
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


def evaluate_predictions(results_df, gt_column, pred_column):
    """takes in the results dataframe, and do analysis on how good my hallucination detection tools are"""

    # Get the ground truth and predictions
    y_true = (df[gt_column] == "PASS").astype(int)
    y_pred = results_df[pred_column].astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Print results
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)


# %% run analysis on ares

df = pd.read_csv("../data/custom_40samples_full_ares_eval.csv")

evaluate_predictions(df, "label", "eval")
