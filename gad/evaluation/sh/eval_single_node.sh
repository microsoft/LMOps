#!/bin/bash

set -x
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name)
            RUN_NAME="$2"
            shift 2
            ;;
        --init_model_path)
            INIT_MODEL_PATH="$2"
            shift 2
            ;;
        --template)
            template="$2"
            shift 2
            ;;
        --tp_size)
            tp_size="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Check required parameters
if [ -z "$RUN_NAME" ] || [ -z "$INIT_MODEL_PATH" ] || [ -z "$template" ] || [ -z "$tp_size" ]; then
    echo "Missing required parameters. Usage:"
    echo "--run_name <run_name> --init_model <init_model> --template <template> --tp_size <tp_size>"
    exit 1
fi

eval_script_path="sh/eval.sh"

HDFS_HOME=TO_BE_DEFINED
CKPT_PATH="checkpoints"
base_checkpoint_path=${HDFS_HOME}/${CKPT_PATH}/${RUN_NAME}

# 定义初始化模型路径, SFT模型
init_model_path=$INIT_MODEL_PATH
chmod +x sh/convert_and_evaluate_gpu.sh

# 调用转化和评估脚本
bash sh/convert_and_evaluate_gpu.sh \
    "$eval_script_path" \
    "$base_checkpoint_path" \
    "$init_model_path" \
    "$template" \
    "$tp_size"




python sh/collect_results.py \
    --base_dir "$base_checkpoint_path/_actor/eval_results" \
    --model_name $init_model_path \
    --wandb_project "openrlhf_ppo_math-eval" \
    --wandb_api_key "${WANDB_API_KEY}" \
    --wandb_run_name $RUN_NAME \
    --benchmarks "gsm8k,math500,minerva_math,gaokao2023en,olympiadbench,college_math,aime24,amc23" \
    --use_wandb 

