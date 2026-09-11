# Learning to Coach for Experiential Learning

This directory contains the implementation of **Learning to Coach (L2C)**.
L2C trains a dedicated LLM-as-a-Coach to turn a frozen actor model's previous
trajectory into concise, actionable experiential knowledge. The coach is
optimized with GRPO using the correctness of the actor's guided solve, while
the actor's parameters remain unchanged.

The implementation is built on [VeRL](https://github.com/volcengine/verl). A
self-contained VeRL snapshot is included under [`verl/`](verl/) so the code
does not depend on a private branch or Git submodule.

After cloning LMOps, run the commands below from this project directory:

```bash
cd LMOps/learn-to-coach
```

## Environment setup

The provided Conda setup targets Linux x86_64 systems with NVIDIA A100, H100,
or H200 GPUs and CUDA 12.4:

```bash
bash tools/setup_conda_env.sh
conda activate l2c
```

Set optional Hugging Face and Weights & Biases credentials through your shell
or secret manager. No credential is stored in this repository.

```bash
export HF_TOKEN="<token-if-needed>"
export WANDB_API_KEY="<key>"
export WANDB_PROJECT=l2c
export L2C_LOGGERS="['console','wandb']"
```

Without `L2C_LOGGERS`, runs log to the console only.

## Data and local paths

The math experiments use the public DAPO train, validation, and test splits.
Download and validate them with:

```bash
bash tools/download_data.sh
```

The scripts use these overridable roots:

```bash
export L2C_DATA_ROOT=/tmp/l2c/data
export L2C_CHECKPOINT_ROOT=/tmp/l2c/checkpoints
export L2C_RESULT_ROOT=/tmp/l2c/results
```

Text-game experiments create deterministic FrozenLake and Sokoban instances
through [TextArena](https://github.com/LeonGuertler/TextArena) and do not need a
separate dataset download.

## L2C protocols

The three paper settings are two orthogonal configuration axes:

| Protocol | Reward scope | Coaching rounds | Actor attempts, `K` | Prompt |
|---|---|---:|---:|---|
| Single-round same-instance | `same_instance` | 1 | 2 | `v5` |
| Iterative same-instance | `same_instance` | 9 | 10 | `v5` |
| Single-round cross-instance | `cross_instance` | 1 | 2 | `v4` |

`coaching_rounds` counts Extract--Guided-solve rounds; the initial solve is
attempt zero, so `K = coaching_rounds + 1`. Cross-instance iterative training
is intentionally unsupported.

For each source instance, training samples eight coach outputs. Same-instance
training rewards each output on another actor attempt on the source. In
cross-instance training, outputs receive their mean accuracy on disjoint
probes. Math uses eight probes shared across the source batch. Text-game uses
eight independent probe seeds per source group, shared by that group's eight
outputs: 64 groups use 512 distinct probe seeds and 4096 probe rollouts per
step. Only the coach is updated. Cross-instance evaluation is separate and
uses 64 sources and one shared pool of 250 probes.

See [`docs/algorithm.md`](docs/algorithm.md) for the dataflow and code map.

## Reproducing the paper configurations

List all provided math and text-game configurations:

```bash
bash usage_example.sh list
```

Train one configuration:

```bash
bash usage_example.sh train math-qwen3-1.7b-same-k2
bash usage_example.sh train math-qwen3-1.7b-same-k10
bash usage_example.sh train frozenlake-qwen3-1.7b-cross-k2
```

The aliases use Hugging Face model IDs by default. Set `MODEL_PATH` to use a
local model directory:

```bash
MODEL_PATH=/models/Qwen3-1.7B \
  bash usage_example.sh train math-qwen3-1.7b-same-k2
```

For a custom run, call the unified driver directly:

```bash
bash scripts/train/train_l2c.sh \
  --model Qwen/Qwen3-1.7B \
  --exp_name my-l2c-run \
  --setting math \
  --reward_scope same_instance \
  --coaching_rounds 1 \
  --source_batch_size 64 \
  --rollout_n 8 \
  --total_training_steps 100
```

Raw Hydra overrides can be appended after `--`. Add `--dry_run` to inspect the
fully resolved command without launching workers.

## Evaluation

Evaluate a saved checkpoint. By default, the helper merges the coach's FSDP
shards to Hugging Face format first, allowing evaluation with a different
world size from training:

```bash
bash usage_example.sh eval math-qwen3-1.7b-same-k2 100
```

Evaluate the identical pipeline with an untrained coach:

```bash
bash usage_example.sh eval-untrained math-qwen3-1.7b-same-k2
```

Evaluate the Self-Refinement baseline:

```bash
bash usage_example.sh self-refine math-qwen3-1.7b-same-k2
```

For text-game Self-Refinement, first create a Phase-A cache and pass its path:

```bash
bash usage_example.sh cache frozenlake-qwen3-1.7b-same-k2
PHASE_A_CACHE=/tmp/l2c/cache/FrozenLake-v0-raw_Qwen3-1.7B_seed0.jsonl \
  bash usage_example.sh self-refine frozenlake-qwen3-1.7b-same-k2
```

The evaluation driver also accepts a native checkpoint or an already merged
coach directly:

```bash
bash scripts/eval/eval_l2c.sh \
  --model Qwen/Qwen3-1.7B \
  --coach_model /path/to/coach-hf \
  --exp_name my-eval \
  --setting math \
  --reward_scope same_instance \
  --coaching_rounds 1
```

Evaluation dumps include the initial solve, extracted knowledge, guided solve,
and summary metrics. Phase-A caches are valid only when the actor, tokenizer,
chat template, dataset order, decoding settings, and seed are unchanged.

## Repository layout

```text
learn-to-coach/
├── README.md
├── usage_example.sh
├── scripts/
│   ├── train/train_l2c.sh
│   └── eval/{eval_l2c.sh,eval_self_refine.sh}
├── tools/
│   ├── setup_conda_env.sh
│   ├── download_data.sh
│   └── merge_model2hf.py
├── docs/algorithm.md
└── verl/                         # vendored VeRL + L2C implementation
```

The central implementation files are:

- `verl/verl/trainer/ppo/l2c_mode.py`: protocol resolution and validation;
- `verl/verl/trainer/ppo/l2c.py`: cross-instance math/text-game reward and eval;
- `verl/verl/trainer/ppo/l2c_train.py`: L2C training orchestration;
- `verl/verl/trainer/ppo/l2c_eval.py`: same-instance, iterative, and Self-Refinement evaluation;
- `verl/verl/trainer/ppo/ray_trainer.py`: Ray workers, resources, and checkpoints;
- `verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`: TextArena interaction.

## Citation

```bibtex
@article{chen2026learningtocoach,
  title={Learning to Coach for Experiential Learning},
  author={Guanheng Chen and Tianzhu Ye and Li Dong and Xun Wu and Shaohan Huang and Furu Wei},
  year={2026}
}
```

## License

The L2C wrapper code is released under the MIT License. The vendored VeRL code
retains its Apache-2.0 license and notice in [`verl/LICENSE`](verl/LICENSE) and
[`verl/Notice.txt`](verl/Notice.txt). The optional lm-eval dependency retains
its upstream MIT License.
