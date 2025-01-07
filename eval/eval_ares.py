# %% Optional for UES/IDP, configure API key for desired model(s)
from dotenv import load_dotenv
import pandas as pd
from ares import ARES

load_dotenv()

# %% process data to feed into ares. first process the few shot samples


# Function to transform the dataframe
def transform_dataframe(df):
    # Rename columns
    df = df.rename(
        columns={
            "question": "Query",
            "passage": "Document",
            "answer": "Answer",
            "label": "Answer_Faithfulness_Label",
        }
    )

    # Transform label values
    df["Answer_Faithfulness_Label"] = df["Answer_Faithfulness_Label"].replace(
        {"FAIL": "[[No]]", "PASS": "[[Yes]]"}
    )

    return df


# Read the CSV files
full_df = pd.read_csv("../data/custom_40samples_full.csv")
fewshot_df = pd.read_csv("../data/custom_16samples_fewshot.csv")

# Transform both dataframes
full_df_transformed = transform_dataframe(full_df)
fewshot_df_transformed = transform_dataframe(fewshot_df)

# Save the transformed dataframes
# need to be explicitly tab separated in ARES
full_df_transformed.to_csv(
    "../data/custom_40samples_full_ares.tsv", sep="\t", index=False
)
fewshot_df_transformed.to_csv(
    "../data/custom_16samples_fewshot_ares.tsv", sep="\t", index=False
)

# Print verification
print("Transformation complete!")
print("\n40 samples dataset columns:", full_df_transformed.columns.tolist())
print(
    "40 samples unique labels:",
    full_df_transformed["Answer_Faithfulness_Label"].unique(),
)
print("\n16 samples dataset columns:", fewshot_df_transformed.columns.tolist())
print(
    "16 samples unique labels:",
    fewshot_df_transformed["Answer_Faithfulness_Label"].unique(),
)

# %% now process the entire dataset


ues_idp_config = {
    "in_domain_prompts_dataset": "../data/custom_16samples_fewshot_ares.tsv",
    "unlabeled_evaluation_set": "../data/custom_40samples_full_ares.tsv",
    "model_choice": "gpt-4o",
}

# uses fewshot examples from IDS
# uses prompt template for CR, AF, AR
# attaches the prompt template and few shot to get the relevant metrics
ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()["Raw Scores"]
print(results)

# note i had to modify the source code to return the raw scores too!

# {'Context Relevance Scores': [Score], 'Answer Faithfulness Scores': [Score], 'Answer Relevance Scores': [Score]}

# %% data processing to save the new csv

full_df = pd.read_csv("../data/custom_40samples_full.csv")

# first assert every result has an evaluation
assert len(results) == len(full_df)

# filter out the hallucinations row
eval_results = results["Answer_Faithfulness_Score"]  # Get the column
eval_results.name = "eval"  # Rename it
merged_df = pd.concat([full_df, eval_results], axis=1)  # Join horizontally

print(merged_df)

# %% save the evaluated results
merged_df.to_csv("../data/custom_40samples_full_ares_eval.csv")

# %%
