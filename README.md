# A Comparison of Hallucination Detection Tools and Research

We benchmark industry hallucination detection tools against each other and current research methods. We use the [HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench) annotated examples to benchmark various hallucination detection tools on these samples, hence a black-box evaluation on prompts and answers only. An overview of observability providers and hallucination detection methods is available [here]()

## Results

| Framework | Accuracy | F1 Score | Precision | Recall |
|-----------|----------|-----------|------------|---------|
| Base (GPT-4o) | 0.754 | 0.760 | 0.742 | 0.778 |
| Base (GPT-4o-mini) | 0.717 | 0.734 | 0.692 | 0.781 |
| Base (3.5sonnet) | 0.790 | 0.803 | 0.758 | 0.854 |
| Base (GPT-4o, sampling) | 0.765 | 0.766 | 0.762 | 0.770 |
| Base (GPT-4o-mini, sampling) | 0.729 | 0.744 | 0.706 | 0.786 |
| CoT (GPT-4o) | **0.833** | **0.831** | **0.840** | 0.822 |
| CoT (GPT-4o-mini) | 0.788 | 0.788 | 0.787 | 0.790 |
| CoT (3.5sonnet) | 0.806 | 0.796 | 0.838 | 0.758 |
| CoT (GPT-4o, sampling) | 0.823 | 0.820 | 0.833 | 0.808 |
| CoT (GPT-4o-mini,sampling) | 0.793 | 0.795 | 0.788 | 0.802 |
| Fewshot (GPT-4o) | 0.737 | 0.773 | 0.680 | **0.896** |
| Lynx | 0.766 | 0.780 | 0.728 | 0.840 |
| RAGAS Faithfulness (GPT-4o) | 0.660 | 0.684 | 0.639 | 0.736 |
| RAGAS Faithfulness (HHEM) | 0.588 | 0.644 | 0.567 | 0.744 |
| G-Eval Hallucination (GPT-4o) | 0.686 | 0.623 | 0.783 | 0.517 |

*Results for the overall dataset*

| Framework | Accuracy | F1 Score | Precision | Recall |
|-----------|----------|-----------|------------|---------|
| Base (GPT-4o) | 0.664 | 0.718 | 0.618 | 0.856 |
| Base (GPT-4o-mini) | 0.594 | 0.698 | 0.556 | 0.940 |
| Base (3.5sonnet) | **0.760** | **0.783** | 0.715 | 0.864 |
| Base (GPT-4o, sampling) | 0.676 | 0.724 | 0.631 | 0.848 |
| Base (GPT-4o-mini, sampling) | 0.616 | 0.714 | 0.569 | 0.960 |
| CoT (GPT-4o) | 0.784 | 0.795 | 0.755 | 0.840 |
| CoT (GPT-4o-mini) | 0.676 | 0.718 | 0.636 | 0.824 |
| CoT (3.5sonnet) | 0.772 | 0.757 | 0.809 | 0.712 |
| CoT (GPT-4o, sampling) | 0.772 | 0.778 | 0.758 | 0.800 |
| CoT (GPT-4o-mini, sampling) | 0.684 | 0.723 | 0.644 | 0.824 |
| Fewshot (GPT-4o) | 0.648 | 0.703 | 0.608 | 0.832 |
| Lynx | 0.724 | 0.760 | 0.673 | 0.872 |
| RAGAS Faithfulness (GPT-4o) | 0.584 | 0.687 | 0.551 | 0.912 |
| RAGAS Faithfulness (HHEM) | 0.552 | 0.687 | 0.528 | **0.984** |
| G-Eval Hallucination (GPT-4o) | 0.681 | 0.592 | **0.829** | 0.460 |

*Results for RAGTruth*

| Framework | Accuracy | F1 Score | Precision | Recall |
|-----------|----------|-----------|------------|---------|
| Base (GPT-4o) | **0.904** | 0.874 | 0.814 | 0.944 |
| Base (GPT-4o-mini) | 0.838 | 0.858 | 0.765 | **0.976** |
| Base (3.5sonnet) | 0.876 | 0.886 | 0.818 | 0.968 |
| Base (GPT-4o, sampling) | **0.904** | **0.902** | 0.816 | 0.920 |
| Base (GPT-4o-mini, sampling) | 0.844 | 0.862 | 0.772 | 0.976 |
| CoT (GPT-4o) | 0.864 | 0.872 | 0.823 | 0.928 |
| CoT (GPT-4o-mini) | 0.860 | 0.869 | 0.817 | 0.928 |
| CoT (3.5sonnet) | 0.868 | 0.873 | 0.843 | 0.904 |
| CoT (GPT-4o, sampling) | 0.852 | 0.863 | 0.801 | 0.936 |
| CoT (GPT-4o-mini, sampling) | 0.864 | 0.874 | 0.814 | 0.944 |
| Fewshot (GPT-4o) | 0.868 | 0.877 | 0.819 | 0.944 |
| Lynx | 0.860 | 0.874 | 0.796 | 0.968 |
| G-Eval Hallucination (GPT-4o) | 0.857 | 0.851 | **0.976** | 0.817 |

*Results for HaluEval*

| Framework | Accuracy | F1 Score | Precision | Recall |
|-----------|----------|-----------|------------|---------|
| Base (GPT-4o) | **0.904** | 0.900 | 0.939 | 0.864 |
| Base (GPT-4o-mini) | 0.824 | 0.829 | 0.807 | 0.852 |
| Base (3.5sonnet) | 0.840 | 0.820 | 0.938 | 0.728 |
| Base (GPT-4o, sampling) | **0.904** | **0.902** | 0.917 | 0.888 |
| Base (GPT-4o-mini, sampling) | 0.840 | 0.845 | 0.820 | 0.872 |
| CoT (GPT-4o) | 0.824 | 0.792 | 0.966 | 0.672 |
| CoT (GPT-4o-mini) | 0.800 | 0.766 | 0.921 | 0.656 |
| CoT (3.5sonnet) | 0.768 | 0.704 | 0.972 | 0.552 |
| CoT (GPT-4o, sampling) | 0.812 | 0.773 | **0.976** | 0.640 |
| CoT (GPT-4o-mini, sampling) | 0.780 | 0.742 | 0.898 | 0.632 |
| Fewshot (GPT-4o) | 0.860 | 0.874 | 0.796 | **0.968** |
| Lynx | 0.800 | 0.806 | 0.782 | 0.832 |
| RAGAS Faithfulness (GPT-4o) | 0.736 | 0.680 | 0.864 | 0.560 |
| RAGAS Faithfulness (HHEM) | 0.624 | 0.573 | 0.663 | 0.504 |
| G-Eval Hallucination (GPT-4o) | 0.600 | 0.333 | 1.000 | 0.200 |

*Results for PubMedQA*

| Framework | Accuracy | F1 Score | Precision | Recall |
|-----------|----------|-----------|------------|---------|
| Base (GPT-4o) | 0.584 | 0.519 | 0.615 | 0.448 |
| Base (GPT-4o-mini) | 0.612 | 0.478 | 0.730 | 0.356 |
| Base (3.5sonnet) | 0.675 | 0.727 | 0.633 | 0.855 |
| Base (GPT-4o, sampling) | 0.624 | 0.530 | 0.707 | 0.424 |
| Base (GPT-4o-mini, sampling) | 0.616 | 0.467 | 0.764 | 0.336 |
| CoT (GPT-4o) | **0.860** | **0.858** | 0.869 | 0.848 |
| CoT (GPT-4o-mini) | 0.816 | 0.803 | 0.862 | 0.752 |
| CoT (3.5sonnet) | 0.816 | 0.824 | 0.788 | **0.864** |
| CoT (GPT-4o, sampling) | 0.856 | 0.856 | 0.856 | 0.856 |
| CoT (GPT-4o-mini, sampling) | 0.844 | 0.838 | **0.871** | 0.808 |
| Fewshot (GPT-4o) | 0.572 | 0.662 | 0.547 | 0.840 |
| Lynx | 0.680 | 0.669 | 0.659 | 0.681 |
| G-Eval Hallucination (GPT-4o) | 0.606 | 0.599 | 0.612 | 0.587 |

*Results for FinanceBench*

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

Combining chain of thought reasoning with sampling, detailed [here](https://arxiv.org/abs/2310.18344). Research done by Galileo AI.

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

## Extra Notes

- Store your environment variables in a `.env` file and use `load_dotenv` to access them later
- We use Jupyter cells within each `py` file to add rerunning of experiments

## Acknowledgments

This project uses code and data from HaluEval[^1].

[^1]: Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models." arXiv preprint arXiv:2305.11747 (2023). https://arxiv.org/abs/2305.11747
