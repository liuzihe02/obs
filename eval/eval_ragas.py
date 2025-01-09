# %%
from datasets import Dataset
import pandas as pd
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    # LLMContextRecall,
    Faithfulness,
    FaithfulnesswithHHEM,
)

# define the llm here
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


# load from csv into ragas dataset for evaluation
def process_data_ragas(dataset_df: pd.DataFrame) -> EvaluationDataset:
    # Use 'Context_Relevance_Label' as 'ground_truth'
    prepared_data = {
        "user_input": dataset_df["question"].tolist(),
        "retrieved_contexts": [[p] for p in dataset_df["passage"].tolist()],
        "response": dataset_df["answer"].tolist(),
        ## no access to ground truth here
        # "ground_truth": dataset_df["label"].tolist(),
    }

    # Convert to HuggingFace's Dataset format
    dataset = Dataset.from_dict(prepared_data)
    # convert to ragas format
    eval_dataset = EvaluationDataset.from_hf_dataset(dataset)
    return eval_dataset


# %% specify metrics
metrics = [
    Faithfulness(llm=evaluator_llm),
    FaithfulnesswithHHEM(),
]
# Load the dataset from the TSV file
eval_df = pd.read_csv("../data/custom_40samples_ragas.csv")
eval_dataset = process_data_ragas(eval_df)
# return the full set of scores for all samples
results = evaluate(dataset=eval_dataset, metrics=metrics)

# %% do some processing into the right data format
results_fth = results["faithfulness"]
# convert to binary labels
results_fth = [1 if score > 0.5 else 0 for score in results_fth]
# convert the list of tensor objects to floats, some versions have tensors some versions have floats
results_fth_hhem = results["faithfulness_with_hhem"]
# also convert to binary labels. 1 for faithful, PASS basically.
results_fth_hhem = [1 if score > 0.5 else 0 for score in results_fth_hhem]

# First create individual dataframes for each eval type
# concat horizontally
ragas_df = pd.concat(
    [
        eval_df["id"],
        pd.DataFrame({"eval_type": "ragas_faithfulness", "eval_result": results_fth}),
    ],
    axis=1,
)
# concat horizontally
ragas_hhem_df = pd.concat(
    [
        eval_df["id"],
        pd.DataFrame(
            {"eval_type": "ragas_faithfulness_hhem", "eval_result": results_fth_hhem}
        ),
    ],
    axis=1,
)

# Concat vertically
final_df = pd.concat(
    [ragas_df, ragas_hhem_df],
    axis=0,
    ignore_index=True,  # only ignore index here if not well lose column headings
)

# %% this results in dataframe with multiple rows for each datapoint.
# each datapoint row is repeated for each eval type
final_df.to_csv("../data/eval_ragas_custom_40samples_ragas.csv")

# %%
