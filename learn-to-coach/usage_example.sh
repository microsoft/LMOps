#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/common.sh
source "${REPO_ROOT}/scripts/common.sh"
cd "${REPO_ROOT}"

list_experiments() {
    printf '%s\n' \
        math-qwen3-1.7b-same-k2 \
        math-qwen3-4b-same-k2 \
        math-qwen3-8b-same-k2 \
        math-qwen3-1.7b-same-k10 \
        math-qwen3-4b-same-k10 \
        math-qwen3-8b-same-k10 \
        math-qwen3-1.7b-cross-k2 \
        math-qwen3-4b-cross-k2 \
        math-qwen3-8b-cross-k2 \
        frozenlake-qwen3-1.7b-same-k2 \
        frozenlake-qwen3-1.7b-same-k10 \
        frozenlake-qwen3-1.7b-cross-k2 \
        sokoban-qwen3-4b-same-k2 \
        sokoban-qwen3-4b-instruct-same-k10 \
        sokoban-qwen3-4b-instruct-cross-k2
}

usage() {
    cat <<'EOF'
Usage:
  bash usage_example.sh list
  bash usage_example.sh train <experiment>
  bash usage_example.sh eval <experiment> <checkpoint-step>
  bash usage_example.sh eval-untrained <experiment>
  bash usage_example.sh cache <experiment>
  bash usage_example.sh self-refine <experiment>

Useful overrides:
  MODEL_PATH=/local/model
  L2C_DATA_ROOT=/path/to/data
  L2C_CHECKPOINT_ROOT=/path/to/checkpoints
  L2C_RESULT_ROOT=/path/to/results
  PHASE_A_CACHE=/path/to/cache.jsonl
EOF
}

set_experiment_metadata() {
    local experiment="$1"

    SOURCE_BATCH_SIZE=64
    ROLLOUT_N=8
    PROBE_SIZE=8
    TEXTGAME_NAME=""
    TEXTGAME_NO_THINK="False"
    NODES=2

    case "${experiment}" in
        math-qwen3-1.7b-same-k2)
            DEFAULT_MODEL="Qwen/Qwen3-1.7B"
            EXP_NAME="math-l2l-q3-1.7b-bs64-binreward-100step-v5-vanilla"
            SETTING=math; REWARD_SCOPE=same_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v5; TRAINING_STEPS=100
            ;;
        math-qwen3-4b-same-k2)
            DEFAULT_MODEL="Qwen/Qwen3-4B"
            EXP_NAME="math-l2l-q3-4b-bs64-binreward-100step-v5-vanilla"
            SETTING=math; REWARD_SCOPE=same_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v5; TRAINING_STEPS=100
            ;;
        math-qwen3-8b-same-k2)
            DEFAULT_MODEL="Qwen/Qwen3-8B"
            EXP_NAME="math-l2l-q3-8b-bs128-binreward-100step-v5-vanilla"
            SETTING=math; REWARD_SCOPE=same_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v5; TRAINING_STEPS=100; SOURCE_BATCH_SIZE=128; NODES=4
            ;;
        math-qwen3-1.7b-same-k10)
            DEFAULT_MODEL="Qwen/Qwen3-1.7B"
            EXP_NAME="math-l2l-q3-1.7b-bs64-iter10-100step-v5"
            SETTING=math; REWARD_SCOPE=same_instance; COACHING_ROUNDS=9
            PROMPT_VERSION=v5; TRAINING_STEPS=100
            ;;
        math-qwen3-4b-same-k10)
            DEFAULT_MODEL="Qwen/Qwen3-4B"
            EXP_NAME="math-l2l-q3-4b-bs64-iter10-100step-v5"
            SETTING=math; REWARD_SCOPE=same_instance; COACHING_ROUNDS=9
            PROMPT_VERSION=v5; TRAINING_STEPS=100
            ;;
        math-qwen3-8b-same-k10)
            DEFAULT_MODEL="Qwen/Qwen3-8B"
            EXP_NAME="math-l2l-q3-8b-bs128-iter10-100step-v5"
            SETTING=math; REWARD_SCOPE=same_instance; COACHING_ROUNDS=9
            PROMPT_VERSION=v5; TRAINING_STEPS=100; SOURCE_BATCH_SIZE=128; NODES=4
            ;;
        math-qwen3-1.7b-cross-k2)
            DEFAULT_MODEL="Qwen/Qwen3-1.7B"
            EXP_NAME="math-l2l-meta-q3-1.7b-bs64-8probe-100step-v4"
            SETTING=math; REWARD_SCOPE=cross_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v4; TRAINING_STEPS=100; NODES=4
            ;;
        math-qwen3-4b-cross-k2)
            DEFAULT_MODEL="Qwen/Qwen3-4B"
            EXP_NAME="math-l2l-meta-q3-4b-bs64-8probe-100step-v4"
            SETTING=math; REWARD_SCOPE=cross_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v4; TRAINING_STEPS=100; NODES=4
            ;;
        math-qwen3-8b-cross-k2)
            DEFAULT_MODEL="Qwen/Qwen3-8B"
            EXP_NAME="math-l2l-meta-q3-8b-bs64-8probe-200step-v4"
            SETTING=math; REWARD_SCOPE=cross_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v4; TRAINING_STEPS=200; NODES=4
            ;;
        frozenlake-qwen3-1.7b-same-k2)
            DEFAULT_MODEL="Qwen/Qwen3-1.7B"
            EXP_NAME="textgame-l2l-q3-1.7b-bs64-binreward-100step-v5-vanilla"
            SETTING=textgame; REWARD_SCOPE=same_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v5; TRAINING_STEPS=100; TEXTGAME_NAME=FrozenLake-v0-raw
            ;;
        frozenlake-qwen3-1.7b-same-k10)
            DEFAULT_MODEL="Qwen/Qwen3-1.7B"
            EXP_NAME="textgame-l2l-q3-1.7b-bs64-binreward-iter10-200step-v5"
            SETTING=textgame; REWARD_SCOPE=same_instance; COACHING_ROUNDS=9
            PROMPT_VERSION=v5; TRAINING_STEPS=200; TEXTGAME_NAME=FrozenLake-v0-raw; NODES=4
            ;;
        frozenlake-qwen3-1.7b-cross-k2)
            DEFAULT_MODEL="Qwen/Qwen3-1.7B"
            EXP_NAME="textgame-l2l-meta-q3-1.7b-bs64-100step-v4-frozenlake"
            SETTING=textgame; REWARD_SCOPE=cross_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v4; TRAINING_STEPS=100; TEXTGAME_NAME=FrozenLake-v0-raw
            NODES=4
            ;;
        sokoban-qwen3-4b-same-k2)
            DEFAULT_MODEL="Qwen/Qwen3-4B"
            EXP_NAME="textgame-l2l-q3-4b-bs64-binreward-100step-v5-vanilla"
            SETTING=textgame; REWARD_SCOPE=same_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v5; TRAINING_STEPS=100; TEXTGAME_NAME=Sokoban-v0; NODES=4
            ;;
        sokoban-qwen3-4b-instruct-same-k10)
            DEFAULT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
            EXP_NAME="textgame-l2l-q3-4b-instruct-2507-bs64-binreward-iter10-100step-v5"
            SETTING=textgame; REWARD_SCOPE=same_instance; COACHING_ROUNDS=9
            PROMPT_VERSION=v5; TRAINING_STEPS=100; TEXTGAME_NAME=Sokoban-v0; NODES=4
            ;;
        sokoban-qwen3-4b-instruct-cross-k2)
            DEFAULT_MODEL="Qwen/Qwen3-4B-Instruct-2507"
            EXP_NAME="textgame-l2l-meta-q3-4b-instruct-2507-bs64-100step-v4-sokoban"
            SETTING=textgame; REWARD_SCOPE=cross_instance; COACHING_ROUNDS=1
            PROMPT_VERSION=v4; TRAINING_STEPS=200; TEXTGAME_NAME=Sokoban-v0
            NODES=4
            ;;
        *)
            echo "Unknown experiment: ${experiment}" >&2
            list_experiments >&2
            exit 2
            ;;
    esac

    MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL}}"
    NNODES="${NNODES:-${NODES}}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
}

run_training() {
    local args=(
        --model "${MODEL_PATH}"
        --exp_name "${EXP_NAME}"
        --setting "${SETTING}"
        --reward_scope "${REWARD_SCOPE}"
        --coaching_rounds "${COACHING_ROUNDS}"
        --prompt_version "${PROMPT_VERSION}"
        --source_batch_size "${SOURCE_BATCH_SIZE}"
        --probe_size "${PROBE_SIZE}"
        --rollout_n "${ROLLOUT_N}"
        --total_training_steps "${TRAINING_STEPS}"
        --nnodes "${NNODES}"
        --gpus_per_node "${GPUS_PER_NODE}"
    )
    if [[ -n "${TEXTGAME_NAME}" ]]; then
        args+=(
            --textgame_name "${TEXTGAME_NAME}"
            --textgame_no_think "${TEXTGAME_NO_THINK}"
        )
    fi
    if l2c_is_true "${L2C_DRY_RUN:-False}"; then
        args+=(--dry_run)
    fi
    bash scripts/train/train_l2c.sh "${args[@]}"
}

eval_common_args() {
    EVAL_ARGS=(
        --model "${MODEL_PATH}"
        --exp_name "${EVAL_NAME}"
        --setting "${SETTING}"
        --reward_scope "${REWARD_SCOPE}"
        --coaching_rounds "${COACHING_ROUNDS}"
        --prompt_version "${PROMPT_VERSION}"
        --nnodes "${EVAL_NNODES:-1}"
        --gpus_per_node "${GPUS_PER_NODE}"
    )
    if [[ "${SETTING}" == "math" ]]; then
        EVAL_ARGS+=(--eval_max_problems 1000)
    else
        EVAL_ARGS+=(
            --textgame_name "${TEXTGAME_NAME}"
            --textgame_no_think "${TEXTGAME_NO_THINK}"
            --held_out_size 500
        )
    fi
    if [[ "${REWARD_SCOPE}" == "cross_instance" ]]; then
        EVAL_ARGS+=(--rollout_n 1 --cross_source_size 64 --cross_probe_size 250)
    elif [[ "${COACHING_ROUNDS}" -gt 1 ]]; then
        EVAL_ARGS+=(--rollout_n 1)
    elif [[ "${SETTING}" == "math" ]]; then
        EVAL_ARGS+=(--rollout_n 16)
    else
        EVAL_ARGS+=(--rollout_n 8)
    fi
    if [[ -n "${PHASE_A_CACHE:-}" ]]; then
        EVAL_ARGS+=(--phase_a_cache "${PHASE_A_CACHE}")
    fi
    if l2c_is_true "${L2C_DRY_RUN:-False}"; then
        EVAL_ARGS+=(--dry_run)
    fi
}

run_trained_eval() {
    local step="$1"
    local checkpoint_root="${L2C_CHECKPOINT_ROOT:-/tmp/l2c/checkpoints}"
    local checkpoint="${checkpoint_root}/${EXP_NAME}/global_step_${step}"
    EVAL_NAME="${EXP_NAME}-eval-step${step}"
    eval_common_args
    EVAL_ARGS+=(--checkpoint "${checkpoint}" --checkpoint_step "${step}")
    if l2c_is_true "${MERGE_TO_HF:-True}"; then
        EVAL_ARGS+=(--merge_to_hf)
    fi
    bash scripts/eval/eval_l2c.sh "${EVAL_ARGS[@]}"
}

run_untrained_eval() {
    EVAL_NAME="${EXP_NAME}-untrained-coach"
    eval_common_args
    bash scripts/eval/eval_l2c.sh "${EVAL_ARGS[@]}"
}

run_cache() {
    local cache_root="${L2C_CACHE_ROOT:-/tmp/l2c/cache}"
    mkdir -p "${cache_root}"
    EVAL_NAME="${EXP_NAME}-phase-a-cache"
    eval_common_args
    if [[ "${SETTING}" == "math" ]]; then
        cache_path="${cache_root}/dapo_test_$(basename -- "${DEFAULT_MODEL}")_seed0.jsonl"
    else
        cache_path="${cache_root}/${TEXTGAME_NAME}_$(basename -- "${DEFAULT_MODEL}")_seed0.jsonl"
    fi
    EVAL_ARGS+=(--phase_a_dump_only --phase_a_cache_output "${cache_path}")
    bash scripts/eval/eval_l2c.sh "${EVAL_ARGS[@]}"
    echo "Phase A cache: ${cache_path}"
}

run_self_refine() {
    local args=(
        --model "${MODEL_PATH}"
        --exp_name "${EXP_NAME}-self-refine"
        --setting "${SETTING}"
        --nnodes "${EVAL_NNODES:-1}"
        --gpus_per_node "${GPUS_PER_NODE}"
    )
    if [[ "${SETTING}" == "math" ]]; then
        args+=(--rollout_n 16 --eval_max_problems 1000)
    else
        [[ -n "${PHASE_A_CACHE:-}" ]] || \
            l2c_die "set PHASE_A_CACHE to a cache produced by 'usage_example.sh cache'"
        args+=(
            --textgame_name "${TEXTGAME_NAME}"
            --textgame_no_think "${TEXTGAME_NO_THINK}"
            --rollout_n 8
            --held_out_size 500
            --phase_a_cache "${PHASE_A_CACHE}"
        )
    fi
    if l2c_is_true "${L2C_DRY_RUN:-False}"; then
        args+=(--dry_run)
    fi
    bash scripts/eval/eval_self_refine.sh "${args[@]}"
}

[[ $# -gt 0 ]] || { usage; exit 2; }
ACTION="$1"
case "${ACTION}" in
    list)
        [[ $# -eq 1 ]] || { usage >&2; exit 2; }
        list_experiments
        ;;
    train|eval-untrained|cache|self-refine)
        [[ $# -eq 2 ]] || { usage >&2; exit 2; }
        set_experiment_metadata "$2"
        case "${ACTION}" in
            train) run_training ;;
            eval-untrained) run_untrained_eval ;;
            cache) run_cache ;;
            self-refine) run_self_refine ;;
        esac
        ;;
    eval)
        [[ $# -eq 3 ]] || { usage >&2; exit 2; }
        set_experiment_metadata "$2"
        run_trained_eval "$3"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
