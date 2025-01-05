# %%
from datasets import load_dataset, Features, Value, ClassLabel
import pandas as pd

# Define the features schema
features = Features(
    {
        "id": Value("string"),
        "passage": Value("string"),
        "question": Value("string"),
        "answer": Value("string"),
        "label": ClassLabel(
            names=["FAIL", "PASS"]
        ),  # Based on the image showing FAIL labels
        "source_ds": ClassLabel(
            names=[
                "halueval",
                "covidQA",
                "pubmedQA",
                "DROP",
                "FinanceBench",
                "RAGTruth",
            ]
        ),  # The 6 values mentioned
    }
)

ds = load_dataset("PatronusAI/HaluBench", features=features)

# %%
ds["test"].features

# %%
