#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="random"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

LLM_MODEL_NAME_OR_PATH="huggyllama/llama-7b"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    LLM_MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/tasks/"
fi

N_SHOTS=8
EVAL_TASKS=("all")

PYTHONPATH=src/ python -u src/inference/generate_few_shot_prompt.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed 1234 \
    --fp16 \
    --llm_eval_tasks "${EVAL_TASKS[@]}" \
    --llm_eval_split test \
    --llm_k_shot "${N_SHOTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}"


PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# EleutherAI/gpt-neo-2.7B # huggyllama/llama-7b
python -u -m torch.distributed.launch --nproc_per_node "${PROC_PER_NODE}" src/main_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed 1234 \
    --fp16 \
    --do_llm_eval \
    --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
    --llm_batch_size_per_device 4 \
    --llm_eval_tasks "${EVAL_TASKS[@]}" \
    --llm_eval_split test \
    --llm_k_shot "${N_SHOTS}" \
    --llm_max_input_length 1024 \
    --llm_max_decode_length 64 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"
