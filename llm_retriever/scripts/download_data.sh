#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

mkdir -p data/tasks/

for DATA_FILE in "train.jsonl.gz" "test.jsonl.gz" "passages.jsonl.gz" "bm25_train.jsonl.gz" "kd_bm25_train.jsonl.gz" "kd_it2_train.jsonl.gz"; do
  if [ ! -e data/tasks/${DATA_FILE} ]; then
    wget -O data/tasks/${DATA_FILE} https://huggingface.co/datasets/intfloat/llm-retriever-tasks/resolve/main/${DATA_FILE}
  fi
done

echo "data downloaded successfully!"
