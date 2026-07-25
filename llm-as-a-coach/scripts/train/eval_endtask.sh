#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=""
EXP_NAME=""
NNODES=1
CKPT=""
USE_BSL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --use_bsl)
            USE_BSL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$EXP_NAME" ] || [ -z "$CKPT" ]; then
    echo "Usage: bash scripts/train/eval_endtask.sh --model PATH --exp_name NAME --ckpt STEP [--use_bsl true]" >&2
    exit 2
fi

EL_CHECKPOINT_ROOT=${EL_CHECKPOINT_ROOT:-/tmp/el/checkpoints}
EL_RESULT_ROOT=${EL_RESULT_ROOT:-/tmp/el/results}
export HF_HOME=${HF_HOME:-/tmp/huggingface}
export TOKENIZERS_PARALLELISM=true

for CKPT_STEP in $CKPT; do
    if [ "$USE_BSL" = "true" ]; then
        final_model_path=$MODEL_PATH
    else
        actor_dir="${EL_CHECKPOINT_ROOT}/${EXP_NAME}/global_step_${CKPT_STEP}/actor"
        model_path="${actor_dir}/huggingface"
        if [ -d "$model_path" ] && [ "$(find "$model_path" -maxdepth 1 -type f -print -quit 2>/dev/null)" ]; then
            echo "Using merged checkpoint: $model_path"
        elif [ -d "$actor_dir" ] && [ "$(find "$actor_dir" -maxdepth 1 -name '*.pt' -type f -print -quit 2>/dev/null)" ]; then
            mkdir -p "$model_path"
            find "$actor_dir" -maxdepth 1 -type f ! -name "*.pt" -exec cp {} "$model_path" \;
            python tools/merge_model2hf.py --local_dir "$actor_dir"
        else
            echo "ERROR: checkpoint not found at ${actor_dir}. Download it before evaluation." >&2
            exit 2
        fi
        final_model_path=$model_path
    fi

    output_dir="${EL_RESULT_ROOT}/${EXP_NAME}/global_step_${CKPT_STEP}/end_task"
    mkdir -p "$output_dir"

    HF_ALLOW_CODE_EVAL=1 lm_eval \
        --model vllm \
        --model_args "pretrained=${final_model_path},enable_thinking=False" \
        --tasks ifeval \
        --batch_size auto \
        --gen_kwargs temperature=0.7,top_p=0.8,top_k=20,min_p=0,do_sample=True \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --write_out \
        --show_config \
        --seed 42 \
        --output_path "$output_dir" \
        --confirm_run_unsafe_code
done
