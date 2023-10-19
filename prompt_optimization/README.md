
# Introduction

This is code for [Automatic Prompt Optimization with "Gradient Descent" and Beam Search](https://arxiv.org/abs/2305.03495) (EMNLP 2023).

The ProTeGi program offers a framework for optimizing and evaluating prompts in text generation tasks. The program supports a variety of evaluation strategies, scoring mechanisms, and is designed to work with tasks that involve binary classification.

The main entrypoint is `main.py`

# Quickstart:
```
time python main.py --task ethos --prompts prompts/ethos.md --data_dir data/ethos --out expt7_datasets/treatment.ucb.ethos.out --evaluator ucb
```

This will run an experiment with UCB bandits for candidate selection. The program will print configuration settings and provide progress updates with each optimization round. The results, including candidate prompts and their associated scores, will be written to the specified output file.

```
python main.py --help
```

For usage instructions. Some of the arguments include:

* `--task`: Task name like 'ethos', 'jailbreak', etc.
* `--data_dir`: Directory where the task data resides.
* `--prompts`: Path to the prompt markdown file.
* `--out`: Output file name.
* `--max_threads`: Maximum number of threads to be used.
* `...`: Various other parameters related to optimization and evaluation.