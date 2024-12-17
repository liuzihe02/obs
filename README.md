# A Comparison of LLM Observability Tools

This project investigates various LLM observability platforms, focusing on issues of reliability and robustness. Specifically, we explore the broad functionalities of various providers, and benchmark their hallucination detection tools against each other and current research methods. We leave in depth investigation of robustness (defense against prompt injections, jailbreaking) to future work.

## Exploration

We investigate the broad functionalities of various providers in `explore/`

## Hallucination Detection

### Benchmarks within Industry

We use the `HaluEval` annotated hallucinations to benchmark various hallucination detection tools on these samples, hence a black-box evaluation on prompts and answers only.

### Benchmarks with Research

We benchmark various industry tools with current hallucination research methods, including grey box and white box methods. Since we will need access to token level log probabilities and activations, we run the user queries in `HaluEval` to generate activations and log probs. We then use these to benchmark both industry and research methods.

## Tips

[To reformat these in `README` properly later. currently these are collected notes/tips]

- Store your environment variables in a `.env` file and use `load_dotenv` to access them later

## Acknowledgments

This project uses code and data from HaluEval[^1].

[^1]: Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models." arXiv preprint arXiv:2305.11747 (2023). https://arxiv.org/abs/2305.11747
