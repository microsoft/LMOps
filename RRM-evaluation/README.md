# Reward Reasoning Model
<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

> This repo is modified based on the [RewardBench](https://github.com/allenai/reward-bench) repository.

## 📦 Model Downloads

<div align="center">

| 🧠 Model   | 🧮 Parameters | 🔗 Download Link  |
|-----------|----------------|------------------|
| RRM-7B     | 7B             | [🤗 HuggingFace](https://huggingface.co/Reward-Reasoning/RRM-7B) |
| RRM-32B    | 32B            | [🤗 HuggingFace](https://huggingface.co/Reward-Reasoning/RRM-32B) |

</div>

---

## 🎯 Pairwise Comparion

This module provides evaluation tools for pairwise comparison tasks using the RRM models. It supports multiple datasets and configurations, enabling fine-grained analysis of model preferences across reasoning challenges.

### 📚 Supported Datasets

| Dataset       | `--dataset` Argument | Uses `round_num` | Valid `round_num` Range |
|---------------|----------------------|------------------|--------------------------|
| RewardBench   | "rewardbench"        | ❌ No            | Any             |
| MMLU-Pro      | "MMLU"               | ✅ Yes           | 1 - 5                    |
| GPQA          | "GPQA"               | ✅ Yes           | 1 - 5                    |
| MATH          | "MATH"               | ✅ Yes           | 1 - 5                    |

For MMLU-Pro, GPQA, and MATH, each `round_num` corresponds to one of five conflict pairs defined in the `PPEdataset` directory.

### 🧪 Usage

Run the evaluation script with:

```bash
export CUDA_VISIBLE_DEVICES=0,1
cd scripts
bash run_7b.sh <seed> <round_num> <dataset>
#e.g.
bash run_7b.sh 0 1 "MATH"
```

### ⚠️ Important Notes
Update the run_name in the corresponding run_7b.sh or run_32b.sh scripts when switching between different datasets. 

We provide example output files for each combination of model_size and dataset in the scripts/results/ directory. These can serve as reference results or templates for further analysis.

## 🎯 ELO Scoring Evaluation
To derive more informative reward signals beyond simply identifying the best response, we adopt an ELO rating system to score each candidate in a round-robin tournament structure. The candidate responses are generated using [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct) on [MATH](https://github.com/hendrycks/math/), provided by the original [Qwen](https://github.com/QwenLM/Qwen2.5-Math) repository.

### 🧪 Usage

Run the evaluation script with:
```bash
cd scripts
python elo_rival_results.py
python elo_accuracy_results.py
```

### ⚠️ Important Notes
Update the filepaths in the corresponding elo_accuracy_result.py and elo_rival_results.py scripts when switching between different models. 