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

    # add the context relevance column and answer relevancy column
    # NOTE: this assumes all the context and answers are relevant; which is probably untrue in practice
    # however we dont extract this so should be fine
    df["Context_Relevance_Label"] = "[[Yes]]"
    df["Answer_Relevance_Label"] = "[[Yes]]"

    return df


# Read the CSV files
full_df = pd.read_csv("../data/custom_16samples.csv")
fewshot_df = pd.read_csv("../data/custom_16samples.csv")

# Transform both dataframes
full_df_transformed = transform_dataframe(full_df)
fewshot_df_transformed = transform_dataframe(fewshot_df)

# Save the transformed dataframes
# need to be explicitly tab separated in ARES
full_df_transformed.to_csv("../data/ares_16samples.tsv", sep="\t", index=False)
fewshot_df_transformed.to_csv("../data/ares_16samples.tsv", sep="\t", index=False)

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
    "in_domain_prompts_dataset": "../data/ares_16samples.tsv",
    "unlabeled_evaluation_set": "../data/ares_16samples.tsv",
    "model_choice": "gpt-4o-mini",
}

# uses fewshot examples from IDS
# uses prompt template for CR, AF, AR
# attaches the prompt template and few shot to get the relevant metrics
ares = ARES(ues_idp=ues_idp_config)
results = ares.ues_idp()["Raw Scores"]
print(results)

# NOTE: i had to modify the source code to return the raw scores too!

# {'Context Relevance Scores': [Score], 'Answer Faithfulness Scores': [Score], 'Answer Relevance Scores': [Score]}

# %% data processing to save the new csv

full_df = pd.read_csv("../data/custom_16samples.csv")

# first assert every result has an evaluation
assert len(results) == len(full_df)

# filter out the hallucinations row
eval_results = results["Answer_Faithfulness_Score"]  # Get the column
eval_results.name = "eval_result"  # Rename it

# Create a new DataFrame with the evaluation type
eval_type = pd.Series(["ares"] * len(full_df), name="eval_type")

merged_df = pd.concat(
    [full_df, eval_results, eval_type], axis=1
)  # Join horizontally without indexing oops

print(merged_df)

# %% save the evaluated results
merged_df.to_csv("../data/eval_ares_16samples.csv")

# %%
