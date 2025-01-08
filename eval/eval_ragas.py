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

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


# load from csv into ragas dataset for evaluation
def load_and_prepare_dataset(file_path: str) -> EvaluationDataset:
    # Load the dataset from the TSV file
    dataset_df = pd.read_csv(file_path)

    # Use 'Context_Relevance_Label' as 'ground_truth'
    prepared_data = {
        "user_input": dataset_df["question"].tolist(),
        "retrieved_contexts": dataset_df["passage"].tolist(),
        "response": dataset_df["answer"].tolist(),
        "ground_truth": dataset_df["label"].tolist(),
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

eval_dataset = load_and_prepare_dataset("../data/custom_16samples_fewshot.csv")
results = evaluate(dataset=eval_dataset, metrics=metrics)
