#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../common.sh
source "${REPO_ROOT}/scripts/common.sh"

MODEL_PATH=""
COACH_MODEL_PATH=""
CHECKPOINT=""
CHECKPOINT_STEP=""
EXP_NAME=""
SETTING="math"
REWARD_SCOPE="same_instance"
COACHING_ROUNDS=1
PROMPT_VERSION=""
NNODES=1
GPUS_PER_NODE=8
ROLLOUT_N=""
EVAL_BATCH_SIZE=""
EVAL_MAX_PROBLEMS=1000
HELD_OUT_SIZE=500
CROSS_SOURCE_SIZE=64
CROSS_PROBE_SIZE=250
PROBE_REWARD_CHUNK_SIZE=""
EXPERIENCE_MAX_LENGTH=8192
MAX_RESPONSE_LENGTH=""
TEXTGAME_NAME="FrozenLake-v0-raw"
TEXTGAME_MAX_STEPS=5
TEXTGAME_NO_THINK="False"
GPU_MEMORY_UTILIZATION=0.8
PPO_MAX_TOKEN_LEN=""
ROLLOUT_SEED=0
PHASE_A_CACHE=""
PHASE_A_DUMP_ONLY="False"
PHASE_A_CACHE_OUTPUT=""
COMPUTE_LOSS_METRICS="False"
MERGE_TO_HF="False"
DATA_ROOT="${L2C_DATA_ROOT:-/tmp/l2c/data}"
RESULT_ROOT="${L2C_RESULT_ROOT:-/tmp/l2c/results}"
DUMP_DIR=""
DRY_RUN="False"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash scripts/eval/eval_l2c.sh --model MODEL --exp_name NAME [options] [-- HYDRA_OVERRIDES...]

Checkpoint options (choose at most one):
  --checkpoint GLOBAL_STEP_DIR   Load native FSDP coach shards
  --coach_model HF_DIR           Load a merged Hugging Face coach
  --merge_to_hf                  Merge --checkpoint/exp_learner before eval
  --checkpoint_step N            Metric/dump step for a merged HF coach

Protocol options:
  --setting math|textgame
  --reward_scope same_instance|cross_instance
  --coaching_rounds N            Paper K = N + 1
  --prompt_version v4|v5
  --rollout_n N
  --eval_max_problems N          Same-instance math
  --held_out_size N              Same-instance text-game
  --cross_source_size N --cross_probe_size N
  --phase_a_cache JSONL
  --phase_a_dump_only --phase_a_cache_output JSONL
  --dump_dir DIR
  --dry_run

With neither --checkpoint nor --coach_model, the script evaluates the
untrained LLM-as-a-Coach initialized from --model.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --coach_model|--exp_model) COACH_MODEL_PATH="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --checkpoint_step) CHECKPOINT_STEP="$2"; shift 2 ;;
        --merge_to_hf) MERGE_TO_HF="True"; shift ;;
        --exp_name|--eval_exp_name) EXP_NAME="$2"; shift 2 ;;
        --setting) SETTING="$2"; shift 2 ;;
        --reward_scope) REWARD_SCOPE="$2"; shift 2 ;;
        --coaching_rounds) COACHING_ROUNDS="$2"; shift 2 ;;
        --prompt_version) PROMPT_VERSION="$2"; shift 2 ;;
        --nnodes) NNODES="$2"; shift 2 ;;
        --gpus_per_node) GPUS_PER_NODE="$2"; shift 2 ;;
        --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
        --eval_batch_size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --eval_max_problems) EVAL_MAX_PROBLEMS="$2"; shift 2 ;;
        --held_out_size) HELD_OUT_SIZE="$2"; shift 2 ;;
        --cross_source_size) CROSS_SOURCE_SIZE="$2"; shift 2 ;;
        --cross_probe_size|--probe_size) CROSS_PROBE_SIZE="$2"; shift 2 ;;
        --probe_reward_chunk_size) PROBE_REWARD_CHUNK_SIZE="$2"; shift 2 ;;
        --experience_max_length) EXPERIENCE_MAX_LENGTH="$2"; shift 2 ;;
        --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
        --textgame_name) TEXTGAME_NAME="$2"; shift 2 ;;
        --textgame_max_steps) TEXTGAME_MAX_STEPS="$2"; shift 2 ;;
        --textgame_no_think) TEXTGAME_NO_THINK="$2"; shift 2 ;;
        --gpu_memory_utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --ppo_max_token_len) PPO_MAX_TOKEN_LEN="$2"; shift 2 ;;
        --rollout_seed) ROLLOUT_SEED="$2"; shift 2 ;;
        --phase_a_cache) PHASE_A_CACHE="$2"; shift 2 ;;
        --phase_a_dump_only) PHASE_A_DUMP_ONLY="True"; shift ;;
        --phase_a_cache_output) PHASE_A_CACHE_OUTPUT="$2"; shift 2 ;;
        --compute_loss_metrics) COMPUTE_LOSS_METRICS="$2"; shift 2 ;;
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
[[ "${REWARD_SCOPE}" == "same_instance" || "${REWARD_SCOPE}" == "cross_instance" ]] || \
    l2c_die "--reward_scope must be same_instance or cross_instance"
[[ "${COACHING_ROUNDS}" =~ ^[1-9][0-9]*$ ]] || l2c_die "--coaching_rounds must be >= 1"
if [[ "${REWARD_SCOPE}" == "cross_instance" && "${COACHING_ROUNDS}" -ne 1 ]]; then
    l2c_die "cross-instance L2C supports exactly one coaching round"
fi
if [[ -n "${CHECKPOINT}" && -n "${COACH_MODEL_PATH}" ]]; then
    l2c_die "--checkpoint and --coach_model are mutually exclusive"
fi
if l2c_is_true "${MERGE_TO_HF}" && [[ -z "${CHECKPOINT}" ]]; then
    l2c_die "--merge_to_hf requires --checkpoint"
fi

if [[ -z "${PROMPT_VERSION}" ]]; then
    if [[ "${REWARD_SCOPE}" == "cross_instance" ]]; then
        PROMPT_VERSION="v4"
    else
        PROMPT_VERSION="v5"
    fi
fi
if [[ -z "${ROLLOUT_N}" ]]; then
    if [[ "${REWARD_SCOPE}" == "cross_instance" || "${COACHING_ROUNDS}" -gt 1 ]]; then
        ROLLOUT_N=1
    elif [[ "${SETTING}" == "math" ]]; then
        ROLLOUT_N=16
    else
        ROLLOUT_N=8
    fi
fi
if [[ -z "${MAX_RESPONSE_LENGTH}" ]]; then
    if [[ "${SETTING}" == "math" ]]; then
        MAX_RESPONSE_LENGTH=16384
    else
        MAX_RESPONSE_LENGTH=1024
    fi
fi
if [[ -z "${EVAL_BATCH_SIZE}" ]]; then
    if [[ "${SETTING}" == "math" ]]; then
        EVAL_BATCH_SIZE=128
    else
        EVAL_BATCH_SIZE=8
    fi
fi

if [[ "${SETTING}" == "math" ]]; then
    if ! l2c_is_true "${DRY_RUN}"; then
        l2c_require_file "${DATA_ROOT}/dapo_train.parquet" "DAPO training data"
        l2c_require_file "${DATA_ROOT}/dapo_test.parquet" "DAPO test data"
    fi
    MAX_PROMPT_LENGTH=$((MAX_RESPONSE_LENGTH + 1024))
else
    MAX_PROMPT_LENGTH=$((EXPERIENCE_MAX_LENGTH + 512 * TEXTGAME_MAX_STEPS))
fi
if [[ -z "${PPO_MAX_TOKEN_LEN}" ]]; then
    PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
fi
if [[ -z "${DUMP_DIR}" ]]; then
    DUMP_DIR="${RESULT_ROOT}/${EXP_NAME}"
fi
mkdir -p "${DUMP_DIR}"

RESUME_MODE="disable"
RESUME_FROM_PATH="null"
if [[ -n "${CHECKPOINT}" ]]; then
    if ! l2c_is_true "${DRY_RUN}"; then
        [[ -d "${CHECKPOINT}/exp_learner" ]] || \
            l2c_die "missing exp_learner checkpoint directory: ${CHECKPOINT}/exp_learner"
    fi
    if [[ -z "${CHECKPOINT_STEP}" && "$(basename -- "${CHECKPOINT}")" =~ ^global_step_([0-9]+)$ ]]; then
        CHECKPOINT_STEP="${BASH_REMATCH[1]}"
    fi
    if l2c_is_true "${MERGE_TO_HF}"; then
        COACH_MODEL_PATH="${CHECKPOINT}/exp_learner/huggingface"
    else
        RESUME_MODE="resume_path"
        RESUME_FROM_PATH="${CHECKPOINT}"
    fi
fi

export NCCL_TIMEOUT="${NCCL_TIMEOUT:-36000}"
export HYDRA_FULL_ERROR=1
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-600}"
export HF_HOME="${HF_HOME:-/tmp/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-l2c-eval}"
LOGGER_BACKENDS="${L2C_LOGGERS:-['console']}"

cmd=(
    python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=grpo
    "trainer.setting=${SETTING}"
    trainer.stage=l2l_eval
    "trainer.l2c_reward_scope=${REWARD_SCOPE}"
    "trainer.l2c_num_coaching_rounds=${COACHING_ROUNDS}"
    "trainer.prompt_version=${PROMPT_VERSION}"
    trainer.exp_sel_with_prev=False
    trainer.max_exp_steps=1
    "trainer.experience_max_length=${EXPERIENCE_MAX_LENGTH}"
    "trainer.resume_mode=${RESUME_MODE}"
    "trainer.resume_from_path=${RESUME_FROM_PATH}"
    "trainer.eval_max_problems=${EVAL_MAX_PROBLEMS}"
    "trainer.held_out_size=${HELD_OUT_SIZE}"
    "trainer.cross_eval_source_size=${CROSS_SOURCE_SIZE}"
    "trainer.cross_eval_probe_size=${CROSS_PROBE_SIZE}"
    "trainer.compute_loss_metrics=${COMPUTE_LOSS_METRICS}"
    data.prompt_key=content
    "data.train_batch_size=${EVAL_BATCH_SIZE}"
    "data.val_batch_size=${EVAL_BATCH_SIZE}"
    "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
    "data.max_response_length=${MAX_RESPONSE_LENGTH}"
    data.truncation=right
    data.validation_shuffle=False
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
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
    "actor_rollout_ref.actor.checkpoint.load_contents=['model','extra']"
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
    trainer.exp_learner_batch_size=1
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
    "trainer.default_local_dir=${RESULT_ROOT}/.eval-checkpoints/${EXP_NAME}"
)

if [[ -n "${COACH_MODEL_PATH}" ]]; then
    cmd+=("actor_rollout_ref.model.exp_model_path=${COACH_MODEL_PATH}")
fi
if [[ -n "${CHECKPOINT_STEP}" && "${RESUME_MODE}" == "disable" ]]; then
    cmd+=("++trainer.force_global_step=${CHECKPOINT_STEP}")
fi
if [[ -n "${PHASE_A_CACHE}" ]]; then
    cmd+=("trainer.phase_a_cache_path=${PHASE_A_CACHE}")
fi
if l2c_is_true "${PHASE_A_DUMP_ONLY}"; then
    cmd+=(trainer.phase_a_dump_only=True)
fi
if [[ -n "${PHASE_A_CACHE_OUTPUT}" ]]; then
    mkdir -p "$(dirname -- "${PHASE_A_CACHE_OUTPUT}")"
    cmd+=("trainer.phase_a_cache_output_path=${PHASE_A_CACHE_OUTPUT}")
fi
if [[ -n "${PROBE_REWARD_CHUNK_SIZE}" ]]; then
    cmd+=("trainer.probe_reward_chunk_size=${PROBE_REWARD_CHUNK_SIZE}")
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
        "trainer.textgame_max_response_length=${MAX_RESPONSE_LENGTH}"
        "trainer.textgame_no_think=${TEXTGAME_NO_THINK}"
        trainer.textgame_total_steps=1
        algorithm.kl_ctrl.kl_coef=0.0
    )
fi
cmd+=("${EXTRA_ARGS[@]}")

PROCESS_RANK="${OMPI_COMM_WORLD_RANK:-${RANK:-0}}"
if [[ "${PROCESS_RANK}" -eq 0 ]]; then
    cd "${REPO_ROOT}"
    if l2c_is_true "${MERGE_TO_HF}"; then
        merge_cmd=(python3 tools/merge_model2hf.py --local_dir "${CHECKPOINT}/exp_learner")
        if [[ -n "$(find "${COACH_MODEL_PATH}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit 2>/dev/null)" ]]; then
            echo "Using existing merged coach: ${COACH_MODEL_PATH}"
        else
            l2c_run_command "${DRY_RUN}" "${merge_cmd[@]}"
        fi
    fi
    l2c_run_command "${DRY_RUN}" "${cmd[@]}"
else
    echo "Rank ${PROCESS_RANK}: waiting while rank 0 owns the Ray driver."
    sleep infinity
fi
