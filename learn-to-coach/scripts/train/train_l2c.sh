#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common.sh
source "${REPO_ROOT}/scripts/common.sh"

MODEL_PATH=""
COACH_MODEL_PATH=""
EXP_NAME=""
SETTING="math"
REWARD_SCOPE="same_instance"
COACHING_ROUNDS=1
PROMPT_VERSION=""
NNODES=1
GPUS_PER_NODE=8
SOURCE_BATCH_SIZE=64
PROBE_SIZE=""
ROLLOUT_N=8
TOTAL_TRAINING_STEPS=100
SAVE_FREQ=5
ACTOR_LR=1e-6
MAX_RESPONSE_LENGTH=""
EXPERIENCE_MAX_LENGTH=8192
TEXTGAME_MAX_RESPONSE_LENGTH=1024
TEXTGAME_NAME="FrozenLake-v0-raw"
TEXTGAME_MAX_STEPS=5
TEXTGAME_NO_THINK="False"
GPU_MEMORY_UTILIZATION=0.8
PPO_MAX_TOKEN_LEN=""
SAVE_OPTIM="True"
AUTO_RESUME="True"
DATA_ROOT="${L2C_DATA_ROOT:-/tmp/l2c/data}"
CHECKPOINT_ROOT="${L2C_CHECKPOINT_ROOT:-/tmp/l2c/checkpoints}"
DUMP_DIR=""
DRY_RUN="False"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/train/train_l2c.sh --model MODEL --exp_name NAME [options] [-- HYDRA_OVERRIDES...]

Core options:
  --setting math|textgame
  --reward_scope same_instance|cross_instance
  --coaching_rounds N        Extract--Guided-solve rounds (paper K = N + 1)
  --prompt_version v4|v5     Defaults to v4 for cross-instance, v5 otherwise
  --coach_model MODEL         Defaults to the actor model
  --source_batch_size N       Number of source instances per optimizer step
  --probe_size N              Cross-instance probes: shared for math, per source group for textgame
  --rollout_n N               Coach candidates per source; must be >= 2
  --total_training_steps N
  --nnodes N --gpus_per_node N
  --checkpoint_root DIR --dump_dir DIR
  --dry_run                   Print the exact command without launching

Text-game options:
  --textgame_name ID
  --textgame_max_steps N
  --textgame_no_think True|False
  --textgame_max_response_length N
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --coach_model|--exp_model) COACH_MODEL_PATH="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --reward_scope) REWARD_SCOPE="$2"; shift 2 ;;
        --coaching_rounds) COACHING_ROUNDS="$2"; shift 2 ;;
        --prompt_version) PROMPT_VERSION="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --gpus_per_node) GPUS_PER_NODE="$2"; shift 2 ;;
        --source_batch_size|--exp_learner_batch_size) SOURCE_BATCH_SIZE="$2"; shift 2 ;;
        --probe_size) PROBE_SIZE="$2"; shift 2 ;;
        --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
        --total_training_steps) TOTAL_TRAINING_STEPS="$2"; shift 2 ;;
        --save_freq) SAVE_FREQ="$2"; shift 2 ;;
        --actor_lr) ACTOR_LR="$2"; shift 2 ;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --experience_max_length) EXPERIENCE_MAX_LENGTH="$2"; shift 2 ;;
        --textgame_max_response_length) TEXTGAME_MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --textgame_name) TEXTGAME_NAME="$2"; shift 2 ;;
        --textgame_max_steps) TEXTGAME_MAX_STEPS="$2"; shift 2 ;;
        --textgame_no_think) TEXTGAME_NO_THINK="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --ppo_max_token_len) PPO_MAX_TOKEN_LEN="$2"; shift 2 ;;
        --save_optim) SAVE_OPTIM="$2"; shift 2 ;;
        --auto_resume) AUTO_RESUME="$2"; shift 2 ;;
        --data_root) DATA_ROOT="$2"; shift 2 ;;
        --checkpoint_root) CHECKPOINT_ROOT="$2"; shift 2 ;;
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
[[ "${REWARD_SCOPE}" == "same_instance" || "${REWARD_SCOPE}" == "cross_instance" ]] || \
    l2c_die "--reward_scope must be same_instance or cross_instance"
[[ "${COACHING_ROUNDS}" =~ ^[1-9][0-9]*$ ]] || l2c_die "--coaching_rounds must be >= 1"
[[ "${ROLLOUT_N}" =~ ^[0-9]+$ ]] || l2c_die "--rollout_n must be an integer"
(( ROLLOUT_N >= 2 )) || l2c_die "--rollout_n must be >= 2 for GRPO"
if [[ "${REWARD_SCOPE}" == "cross_instance" && "${COACHING_ROUNDS}" -ne 1 ]]; then
    l2c_die "cross-instance L2C supports exactly one coaching round"
fi

if [[ -z "${PROMPT_VERSION}" ]]; then
    if [[ "${REWARD_SCOPE}" == "cross_instance" ]]; then
        PROMPT_VERSION="v4"
    else
        PROMPT_VERSION="v5"
    fi
fi
if [[ -z "${PROBE_SIZE}" ]]; then
    PROBE_SIZE=8
fi
if [[ -z "${MAX_RESPONSE_LENGTH}" ]]; then
    if [[ "${SETTING}" == "math" ]]; then
        MAX_RESPONSE_LENGTH=16384
    else
        MAX_RESPONSE_LENGTH=8192
    fi
fi

if [[ "${SETTING}" == "math" ]]; then
    if ! l2c_is_true "${DRY_RUN}"; then
        l2c_require_file "${DATA_ROOT}/dapo_train.parquet" "DAPO training data"
        l2c_require_file "${DATA_ROOT}/dapo_test.parquet" "DAPO test data"
    fi
    MAX_PROMPT_LENGTH=$((MAX_RESPONSE_LENGTH + 1024))
    if [[ "${REWARD_SCOPE}" == "cross_instance" ]]; then
        TRAIN_BATCH_SIZE=$((SOURCE_BATCH_SIZE + PROBE_SIZE))
    else
        TRAIN_BATCH_SIZE=${SOURCE_BATCH_SIZE}
    fi
else
    MAX_PROMPT_LENGTH=$((EXPERIENCE_MAX_LENGTH + 512 * TEXTGAME_MAX_STEPS))
    TRAIN_BATCH_SIZE=${SOURCE_BATCH_SIZE}
fi
if [[ -z "${PPO_MAX_TOKEN_LEN}" ]]; then
    PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
fi

LOCAL_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}"
mkdir -p "${LOCAL_DIR}"
if [[ -n "${DUMP_DIR}" ]]; then
    mkdir -p "${DUMP_DIR}"
fi

if l2c_is_true "${SAVE_OPTIM}"; then
    SAVE_CONTENTS="['model','extra','optimizer']"
else
    SAVE_CONTENTS="['model','extra']"
fi
if l2c_is_true "${AUTO_RESUME}"; then
    RESUME_MODE="auto"
else
    RESUME_MODE="disable"
fi

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-36000}"
export HYDRA_FULL_ERROR=1
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-l2c}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
LOGGER_BACKENDS="${L2C_LOGGERS:-['console']}"

cmd=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    "trainer.setting=${SETTING}"
    trainer.stage=l2l
    "trainer.l2c_reward_scope=${REWARD_SCOPE}"
    "trainer.l2c_num_coaching_rounds=${COACHING_ROUNDS}"
    "trainer.prompt_version=${PROMPT_VERSION}"
    trainer.exp_sel_with_prev=False
    trainer.max_exp_steps=1
    "trainer.experience_max_length=${EXPERIENCE_MAX_LENGTH}"
    "data.prompt_key=content"
    "data.train_batch_size=${TRAIN_BATCH_SIZE}"
    data.val_batch_size=1
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.truncation=right
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR}"
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
    "actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS}"
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.temperature=0.6
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=20
    "trainer.exp_learner_batch_size=${SOURCE_BATCH_SIZE}"
    "trainer.probe_size=${PROBE_SIZE}"
    trainer.val_before_train=False
    trainer.critic_warmup=0
    "trainer.logger=${LOGGER_BACKENDS}"
    "trainer.project_name=${WANDB_PROJECT}"
    "trainer.experiment_name=${EXP_NAME}"
    "trainer.n_gpus_per_node=${GPUS_PER_NODE}"
    "trainer.nnodes=${NNODES}"
    "trainer.save_freq=${SAVE_FREQ}"
    trainer.test_freq=100000000
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
    trainer.total_epochs=3000000000000000000
    trainer.default_hdfs_dir=null
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_sleep_hack=True
    "trainer.dump_dir=${DUMP_DIR:-null}"
    "trainer.default_local_dir=${LOCAL_DIR}"
    "trainer.resume_mode=${RESUME_MODE}"
)

if [[ -n "${COACH_MODEL_PATH}" && "${COACH_MODEL_PATH}" != "${MODEL_PATH}" ]]; then
    cmd+=("actor_rollout_ref.model.exp_model_path=${COACH_MODEL_PATH}")
fi
if [[ "${SETTING}" == "math" ]]; then
    cmd+=(
        "data.train_files=${DATA_ROOT}/dapo_train.parquet"
        "data.val_files=${DATA_ROOT}/dapo_test.parquet"
    )
else
    cmd+=(
        "trainer.textgame_env_id=${TEXTGAME_NAME}"
        "trainer.textgame_max_steps=${TEXTGAME_MAX_STEPS}"
        trainer.textgame_wfeedback=True
        trainer.textgame_keep_reasoning=True
        "trainer.textgame_max_prompt_length=${MAX_PROMPT_LENGTH}"
        "trainer.textgame_max_response_length=${TEXTGAME_MAX_RESPONSE_LENGTH}"
        "trainer.textgame_no_think=${TEXTGAME_NO_THINK}"
        "trainer.textgame_total_steps=${TOTAL_TRAINING_STEPS}"
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
