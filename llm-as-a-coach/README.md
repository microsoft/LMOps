# LLM-as-a-Coach: Experiential Learning for Non-Verifiable Tasks

This repository contains the implementation for 📄 **[LLM-as-a-Coach: Experiential Learning for Non-Verifiable Tasks](https://arxiv.org/html/2607.18110v1)**.

Experiential Learning (EL) turns an LLM-as-a-Judge into an LLM-as-a-Coach. Instead of reducing feedback to a scalar reward, the coach extracts textual experiential knowledge from on-policy responses. A teacher conditions on that knowledge, and the policy internalizes it through on-policy context distillation.

The code is built on [VeRL](https://github.com/volcengine/verl).

## 🚀 Environment Setup

The Conda setup supports Linux x86_64 systems with NVIDIA A100, H100, or H200 GPUs. Run it from the repository root, then activate the environment:

```bash
bash tools/setup_conda_env.sh
conda activate el
```

To use another environment name, pass it as the first argument, for example `bash tools/setup_conda_env.sh my-el-env`.

IFEval additionally requires [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
pip install lm-eval langdetect immutabledict
```

Set your Weights & Biases credentials in the environment before training. `HF_TOKEN` is optional and is read directly by Hugging Face libraries when needed.

```bash
export WANDB_PROJECT=el
# Export WANDB_API_KEY through your shell or secret manager.
# Export HF_TOKEN as well if the model or dataset is gated.
```

Experiments using GPT-4o as the coach or reward model use the standard OpenAI Python SDK. Export `OPENAI_API_KEY`; the SDK reads it automatically. `OPENAI_BASE_URL` is optional for an OpenAI-compatible endpoint, and `OPENAI_MODEL` defaults to `gpt-4o`.

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_MODEL=gpt-4o
# export OPENAI_BASE_URL="https://api.openai.com/v1"
```

## 📦 Data and checkpoints

All local paths default to `/tmp/el` and can be overridden without editing the scripts:

```bash
export EL_DATA_ROOT=/tmp/el/data
export EL_CHECKPOINT_ROOT=/tmp/el/checkpoints
export EL_RESULT_ROOT=/tmp/el/results
```

Download all required training and evaluation data from [ytz20/el-data](https://huggingface.co/datasets/ytz20/el-data):

```bash
bash tools/download_data.sh
```

The script downloads to `EL_DATA_ROOT` (`/tmp/el/data` by default), preserves the expected directory layout, and verifies that every required file is present. A different repository or destination can be supplied when needed:

```bash
bash tools/download_data.sh <namespace/dataset-repo> <data-dir>
```

The resulting layout is:

```text
/tmp/el/data/
├── wildchat-if_rubric-4o_train.parquet
├── wildchat-if_rubric-4o_val.parquet
├── tulu-3-sft-mixture-filtered.parquet
├── alpacaeval2/alpaca_eval_gpt4_baseline.json
├── wildbench/v2.json
├── arena_hard_v2/prompts.json
└── creativewritingv3/creative_writing_prompts_v3.json
```

Training checkpoints are written to `/tmp/el/checkpoints/<experiment>/global_step_<step>`. Evaluation scripts read from the same location, merge an FSDP actor checkpoint when necessary, and write results under `/tmp/el/results`.

## 📖 Experiments

The paper configurations cover Qwen3-8B and OLMo-3-7B-Instruct, using either the policy itself or GPT-4o for feedback. The complete commands are in [`usage_example.sh`](usage_example.sh).

<details>
<summary>Train</summary>

```bash
bash usage_example.sh list

bash usage_example.sh train qwen-el-self
bash usage_example.sh train qwen-el-self-iter
bash usage_example.sh train qwen-el-gpt4o
bash usage_example.sh train olmo-el-self
bash usage_example.sh train olmo-el-gpt4o
bash usage_example.sh train qwen-rl-self
bash usage_example.sh train qwen-rl-gpt4o
bash usage_example.sh train olmo-rl-self
bash usage_example.sh train olmo-rl-gpt4o
```

</details>

<details>
<summary>Eval</summary>

Each experiment supports the held-out WildChat evaluation, the fuzzy-task generation suite, and IFEval. Supply the checkpoint step explicitly and replace `qwen-el-self` with any experiment printed by `bash usage_example.sh list`.

```bash
bash usage_example.sh eval qwen-el-self 80
bash usage_example.sh eval_fuzzy qwen-el-self 80
bash usage_example.sh eval_endtask qwen-el-self 80
```

`eval` and `eval_fuzzy` generate model responses. `eval_endtask` runs and scores IFEval directly.

</details>

<details>
<summary>Score</summary>

After `eval` and `eval_fuzzy` finish, run the GPT-4o evaluators to obtain the final scores. The scoring scripts take the experiment's checkpoint-directory name rather than the short alias used by `usage_example.sh`:

```bash
# qwen-el-self example
EXP_NAME=wildchat-el-q3-8b-r8b
CKPT=80

python scripts/eval/eval_gpt4o.py \
    --exp_name "$EXP_NAME" \
    --start_ckpt "$CKPT" \
    --end_ckpt "$CKPT"

python scripts/eval/eval_gpt4o_fuzzy.py \
    --benchmark alpacaeval2,wildbench,arena_hard_v2,creativewritingv3 \
    --exp_name "$EXP_NAME" \
    --start_ckpt "$CKPT" \
    --end_ckpt "$CKPT"
```

Set `OPENAI_API_KEY` as described above before scoring. Fuzzy evaluation scores full, unshortened responses by default. For multiple checkpoints, set `--start_ckpt`, `--end_ckpt`, and `--step` accordingly. IFEval reports its score directly and does not use either GPT-4o evaluator.

</details>

## 📄 Citation

```bibtex
@article{ye2026llmasacoach,
  title={LLM-as-a-Coach: Experiential Learning for Non-Verifiable Tasks},
  author={Tianzhu Ye and Li Dong and Guanheng Chen and He Zhu and Xun Wu and Shaohan Huang and Furu Wei},
  year={2026},
  eprint={2607.18110},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2607.18110}
}
```
