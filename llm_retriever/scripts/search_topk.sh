#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="bm25"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

SPLIT="train"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    SPLIT=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/tasks/"
fi

PYTHONPATH=src/ python -u src/inference/search_topk.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_search \
    --fp16 \
    --search_split "${SPLIT}" \
    --search_topk 100 \
    --dataloader_num_workers 1 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"
