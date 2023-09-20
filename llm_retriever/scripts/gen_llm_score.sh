#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

# EleutherAI/gpt-neo-2.7B
MODEL_NAME_OR_PATH="huggyllama/llama-7b"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

SPLIT="bm25_train"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/tasks/"
fi

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
# For gpt-neo-2.7B, set batch_size_per_device to 16
# llama-7b, set batch_size_per_device to 8
PYTHONPATH=src/ python -u -m torch.distributed.launch --nproc_per_node ${PROC_PER_NODE} src/inference/gen_llm_scores.py \
    --llm_model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --fp16 \
    --search_split "${SPLIT}" \
    --search_topk 32 \
    --llm_batch_size_per_device 8 \
    --max_train_samples 200000 \
    --output_dir "/tmp/" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"
