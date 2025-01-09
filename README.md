# A Comparison of Hallucination Detection Tools and Research

We benchmark industry hallucination detection tools against each other and current research methods. We use the [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench) annotated examples to benchmark various hallucination detection tools on these samples, hence a black-box evaluation on prompts and answers only.

## Results

| Framework | Accuracy | F1 Score | Precision | Recall |
|-----------|----------|-----------|------------|---------|
| LLM-Judge *(GPT-4o)* | 0.89 | 0.86 | 0.88 | 0.84 |
| LLM-Judge *(GPT-4o-mini)* | 0.89 | 0.86 | 0.88 | 0.84 |
| LLM-Judge, Sampling *(GPT-4o)* | 0.89 | 0.86 | 0.88 | 0.84 |
| LLM-Judge, CoT *(GPT-4o)* | 0.89 | 0.86 | 0.88 | 0.84 |
| ChainPoll *(GPT-4o)* | 0.87 | 0.85 | 0.86 | 0.84 |
| ARES *(GPT-4o)* | 0.92 | 0.90 | 0.91 | 0.89 |
| RAGAS Faithfulness *(GPT-4o)* | 0.85 | 0.83 | 0.84 | 0.82 |
| RAGAS Faithfulness *(HHEM)* | 0.85 | 0.83 | 0.84 | 0.82 |
| G-Eval *(GPT-4o)* | 0.90 | 0.88 | 0.89 | 0.87 |
| Lynx | 0.86 | 0.84 | 0.85 | 0.83 |

*Results for the overall dataset*

**Comparisons**
- LLM-Judge
- ARES(Few-shot w/GPT4, GPT4o)
- RAGAS
- ChainPoll
- G-Eval
- Lynx

**To-Do**
- Self Consistency (Mundler)
- LUNA
- SelfCheckGPT


## HaluBench Dataset

The HaluBench dataset consist of `question,passage,answer` triplets and a `label` indicating if the answer to the question was faithful to the passage. `"PASS"==1`, `"FAIL"==0`, hence `PASS` indicates no hallucinations, whereas `FAIL` indicates hallucinations were present.

- We use RAGTruth, HaluEval, and PubMedQA, and FinanceBench
- Each kind of dataset takes equal proportion of total dataset
- Within each kind of dataset, 50% `PASS` and 50% `FAIL`


## Methods

### LLM-Judge

 We use both `openai` and `anthropic` models.

> We use a similar prompt template to `arize` phoenix hallucination detection.

**Zero-Shot**

We simply pass in the entire triplet to an LLM and ask it to detect hallucinations.

**Chain-of-Thought**

We modify the prompt such that before generating the answer, the LLM comes up with an explanation to aid its answer.

**Few-Shot**

We concatenate a few labelled examples to the prompt.

**Sampling**

We repeat the same prompt 5 times and take the average score.

**ChainPoll**

Combining chain of thought reasoning with sampling, detailed [here](https://arxiv.org/abs/2310.18344)

### ARES

As we don't have access to the fine-tuned [ARES](https://github.com/stanford-futuredata/ARES) LLM-Judges (particularly for answer faithfulness), we simply use the ARES prompt for hallucination detection, similar to LLM-Judge.

- We modify the `score_row` function in `ues_idp` to remove the extra calls for `context_relevance` `answer_relevance`. This because we only detect intrinsic hallucinations, which are based on `answer_faithfulness` only
- We modify `ues_idp_config` to return the results for each datapoint

### RAGAS

We use [RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) faithfulness and also faithfulness with HHEM

> Answer faithfulness metric doesn't work well for single word answers, which comprise a huge portion of data
> We only use the subset of the datasets RAGTruth and PubMedQA for this, as these datasets are longform contexts.

### G-Eval

We use the [G-Eval](https://docs.confident-ai.com/docs/metrics-llm-evals) framework here, which uses chain of thought prompting and taking the probabilities of the output logits for the metric

### Lynx

The [Lynx](https://arxiv.org/abs/2407.08488) model is a fine-tuned verison of `Llama-3-70BInstruct` on the `HaluBench` dataset.

> This model performs well due to it being explicitly fine-tuned on this dataset

## Tips

[To reformat these in `README` properly later. currently these are collected notes/tips]

- Store your environment variables in a `.env` file and use `load_dotenv` to access them later

## Acknowledgments

This project uses code and data from HaluEval[^1].

[^1]: Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models." arXiv preprint arXiv:2305.11747 (2023). https://arxiv.org/abs/2305.11747
