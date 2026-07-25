#!/bin/bash
# set -x

export NCCL_TIMEOUT=36000
# Parse command-line arguments
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
            break
            ;;
    esac
done

USE_BSL=${USE_BSL:-false}
EL_CHECKPOINT_ROOT=${EL_CHECKPOINT_ROOT:-/tmp/el/checkpoints}
EL_RESULT_ROOT=${EL_RESULT_ROOT:-/tmp/el/results}

export HF_HOME=${HF_HOME:-/tmp/huggingface}
export TOKENIZERS_PARALLELISM=true

BENCHMARKS=(alpacaeval2 wildbench arena_hard_v2 creativewritingv3)

# Handle multiple checkpoints if provided
CKPT_LIST=$CKPT
for CKPT in $CKPT_LIST; do
    echo "=========================================="
    echo "Processing Checkpoint: $CKPT"
    echo "=========================================="

    # merge checkpoint
    if [ "$USE_BSL" = "true" ]; then
        echo "Skipping model merge as USE_BSL is true"
        final_model_path=$MODEL_PATH
    else
        resume_dir="${EL_CHECKPOINT_ROOT}/${EXP_NAME}/global_step_${CKPT}/actor"
        model_path="${resume_dir}/huggingface"
        if [ -d "$model_path" ] && [ "$(find "$model_path" -maxdepth 1 -type f -print -quit 2>/dev/null)" ]; then
            echo "Huggingface directory already exists and has files, skipping model preparation: $model_path"
        elif [ -d "$resume_dir" ] && [ "$(find "$resume_dir" -maxdepth 1 -name "*.pt" -type f -print -quit 2>/dev/null)" ]; then
            mkdir -p "$model_path"
            find "$resume_dir" -maxdepth 1 -type f ! -name "*.pt" -exec cp {} "$model_path" \;
            python tools/merge_model2hf.py --local_dir "$resume_dir"
        else
            echo "ERROR: checkpoint not found at ${resume_dir}. Download it before evaluation." >&2
            exit 2
        fi
        final_model_path=$model_path
    fi

    OUTPUT_DIR=${EL_RESULT_ROOT}/${EXP_NAME}/global_step_${CKPT}/eval_fuzzy

    for bench in "${BENCHMARKS[@]}"; do
        echo "Running generation for benchmark: $bench"
        python scripts/eval/gen_fuzzy.py \
            --benchmark $bench \
            --model $final_model_path \
            --output_dir $OUTPUT_DIR \
            --temperature 0.7 \
            --top_p 0.95 \
            --max_tokens 4096
    done
done
