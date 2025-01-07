# A Comparison of Hallucination Detection Tools and Research

We benchmark industry hallucination detection tools against each other and current research methods.

## Hallucination Detection Benchmarks

We use the `HaluEval` annotated hallucinations to benchmark various hallucination detection tools on these samples, hence a black-box evaluation on prompts and answers only.

**Comparisons**
- ARES
  - Few-shot w/GPT4, GPT4o
  - Zero-shot w/GPT4, GPT4o
- RAGAS
- SelfCheckGPT
- G-Eval
- Lynx

- Self Consistency (Mundler)
- LUNA
- ChainPoll

### HaluBench Dataset

- We use RAGTruth, HaluEval, and PubMedQA, and FinanceBench
- Each kind of dataset takes equal proportion of total dataset
- Within each kind of dataset, 50% `PASS` and 50% `FAIL`
  - `"PASS"==1`, `"FAIL"==0`

### LLM-Judge

We use both `openai` and `anthropic` models.

**Zero-Shot**

**Chain-of-Thought**

**Few-Shot**

### ARES

- We modify the `score_row` function in `ues_idp` to remove the extra calls for `context_relevance` `answer_relevance`. This because we only detect intrinsic hallucinations, which are based on `answer_faithfulness` only
- We modify `ues_idp_config` to return the results for each datapoint

> The checkpoint for answer faithfulness isn't provided

### RAGAS

> Answer faithfulness metric doesn't work well for single word answers, which comprise a huge portion of data

## Tips

[To reformat these in `README` properly later. currently these are collected notes/tips]

- Store your environment variables in a `.env` file and use `load_dotenv` to access them later

## Acknowledgments

This project uses code and data from HaluEval[^1].

[^1]: Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models." arXiv preprint arXiv:2305.11747 (2023). https://arxiv.org/abs/2305.11747
