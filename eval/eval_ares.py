# %% Optional for UES/IDP, configure API key for desired model(s)
from dotenv import load_dotenv
import pandas as pd

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
df_40 = pd.read_csv("data/custom_40samples.csv")
df_16 = pd.read_csv("data/custom_16samples.csv")

# Transform both dataframes
df_40_transformed = transform_dataframe(df_40)
df_16_transformed = transform_dataframe(df_16)

# Save the transformed dataframes
df_40_transformed.to_csv("data/custom_40samples_transformed.csv", index=False)
df_16_transformed.to_csv("data/custom_16samples_transformed.csv", index=False)

# Print verification
print("Transformation complete!")
print("\n40 samples dataset columns:", df_40_transformed.columns.tolist())
print(
    "40 samples unique labels:", df_40_transformed["Answer_Faithfulness_Label"].unique()
)
print("\n16 samples dataset columns:", df_16_transformed.columns.tolist())
print(
    "16 samples unique labels:", df_16_transformed["Answer_Faithfulness_Label"].unique()
)

# %% now process the entire dataset
