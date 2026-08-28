#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common.sh
source "${REPO_ROOT}/scripts/common.sh"

MODEL_PATH=""
EXP_NAME=""
SETTING="math"
NNODES=1
GPUS_PER_NODE=8
ROLLOUT_N=""
ROLLOUT_SEED=0
EVAL_BATCH_SIZE=128
EVAL_MAX_PROBLEMS=1000
HELD_OUT_SIZE=500
MAX_RESPONSE_LENGTH=""
EXPERIENCE_MAX_LENGTH=8192
TEXTGAME_NAME="FrozenLake-v0-raw"
TEXTGAME_MAX_STEPS=5
TEXTGAME_NO_THINK="False"
PHASE_A_CACHE=""
GPU_MEMORY_UTILIZATION=0.8
DATA_ROOT="${L2C_DATA_ROOT:-/tmp/l2c/data}"
RESULT_ROOT="${L2C_RESULT_ROOT:-/tmp/l2c/results}"
DUMP_DIR=""
DRY_RUN="False"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/eval/eval_self_refine.sh --model MODEL --exp_name NAME [options]

  --setting math|textgame
  --rollout_n N
  --eval_max_problems N          Math
  --textgame_name ID             Text-game
  --held_out_size N              Text-game
  --phase_a_cache JSONL          Required for text-game self-refinement
  --dry_run
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --exp_name|--eval_exp_name) EXP_NAME="$2"; shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --gpus_per_node) GPUS_PER_NODE="$2"; shift 2 ;;
        --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
        --rollout_seed) ROLLOUT_SEED="$2"; shift 2 ;;
        --eval_batch_size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --eval_max_problems) EVAL_MAX_PROBLEMS="$2"; shift 2 ;;
        --held_out_size) HELD_OUT_SIZE="$2"; shift 2 ;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --experience_max_length) EXPERIENCE_MAX_LENGTH="$2"; shift 2 ;;
        --textgame_name) TEXTGAME_NAME="$2"; shift 2 ;;
        --textgame_max_steps) TEXTGAME_MAX_STEPS="$2"; shift 2 ;;
        --textgame_no_think) TEXTGAME_NO_THINK="$2"; shift 2 ;;
        --phase_a_cache) PHASE_A_CACHE="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --result_root) RESULT_ROOT="$2"; shift 2 ;;
        --dump_dir) DUMP_DIR="$2"; shift 2 ;;
        --dry_run) DRY_RUN="True"; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) l2c_die "unknown option '$1' (put raw Hydra overrides after --)" ;;
    esac
done

[[ -n "${MODEL_PATH}" ]] || l2c_die "--model is required"
[[ -n "${EXP_NAME}" ]] || l2c_die "--exp_name is required"
[[ "${SETTING}" == "math" || "${SETTING}" == "textgame" ]] || \
    l2c_die "--setting must be math or textgame"

if [[ "${SETTING}" == "math" ]]; then
    ROLLOUT_N="${ROLLOUT_N:-16}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
    MAX_PROMPT_LENGTH=$((MAX_RESPONSE_LENGTH + 1024))
    if ! l2c_is_true "${DRY_RUN}"; then
        l2c_require_file "${DATA_ROOT}/dapo_train.parquet" "DAPO training data"
        l2c_require_file "${DATA_ROOT}/dapo_test.parquet" "DAPO test data"
    fi
    STAGE="self_refine_eval"
else
    ROLLOUT_N="${ROLLOUT_N:-8}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
    MAX_PROMPT_LENGTH=$((EXPERIENCE_MAX_LENGTH + 512 * TEXTGAME_MAX_STEPS))
    [[ -n "${PHASE_A_CACHE}" ]] || l2c_die "--phase_a_cache is required for textgame"
    if ! l2c_is_true "${DRY_RUN}"; then
        l2c_require_file "${PHASE_A_CACHE}" "Phase A cache"
    fi
    STAGE="self_refine_eval_textgame"
fi
PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
if [[ -z "${DUMP_DIR}" ]]; then
    DUMP_DIR="${RESULT_ROOT}/${EXP_NAME}"
fi
mkdir -p "${DUMP_DIR}"

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-36000}"
export HYDRA_FULL_ERROR=1
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-l2c-eval}"
LOGGER_BACKENDS="${L2C_LOGGERS:-['console']}"

cmd=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    "trainer.stage=${STAGE}"
    "trainer.setting=${SETTING}"
    trainer.resume_mode=disable
    "trainer.eval_max_problems=${EVAL_MAX_PROBLEMS}"
    "trainer.held_out_size=${HELD_OUT_SIZE}"
    data.prompt_key=content
    "data.train_batch_size=${EVAL_BATCH_SIZE}"
    "data.val_batch_size=${EVAL_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.truncation=right
    data.validation_shuffle=False
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=128000
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}"
    "actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN}"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.temperature=0.6
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    "++actor_rollout_ref.rollout.seed=${ROLLOUT_SEED}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    trainer.val_before_train=False
    trainer.critic_warmup=0
    "trainer.logger=${LOGGER_BACKENDS}"
    "trainer.project_name=${WANDB_PROJECT}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    trainer.save_freq=10000000000
    trainer.test_freq=10000000000
    trainer.total_training_steps=1
    trainer.total_epochs=1
    trainer.default_hdfs_dir=null
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_sleep_hack=True
    "trainer.dump_dir=${DUMP_DIR}"
    "trainer.default_local_dir=${RESULT_ROOT}/.self-refine/${EXP_NAME}"
)

if [[ "${SETTING}" == "math" ]]; then
    cmd+=(
        "data.train_files=${DATA_ROOT}/dapo_train.parquet"
        "data.val_files=${DATA_ROOT}/dapo_test.parquet"
    )
else
    cmd+=(
        "trainer.textgame_env_id=${TEXTGAME_NAME}"
        "trainer.textgame_max_steps=${TEXTGAME_MAX_STEPS}"
        trainer.textgame_keep_reasoning=True
        "trainer.textgame_no_think=${TEXTGAME_NO_THINK}"
        "trainer.textgame_max_prompt_length=${MAX_PROMPT_LENGTH}"
        "trainer.textgame_max_response_length=${MAX_RESPONSE_LENGTH}"
        trainer.textgame_wfeedback=True
        trainer.textgame_total_steps=1
        "trainer.phase_a_cache_path=${PHASE_A_CACHE}"
    )
fi
cmd+=("${EXTRA_ARGS[@]}")

PROCESS_RANK="${OMPI_COMM_WORLD_RANK:-${RANK:-0}}"
if [[ "${PROCESS_RANK}" -eq 0 ]]; then
    cd "${REPO_ROOT}"
    l2c_run_command "${DRY_RUN}" "${cmd[@]}"
else
    echo "Rank ${PROCESS_RANK}: waiting while rank 0 owns the Ray driver."
    sleep infinity
fi
