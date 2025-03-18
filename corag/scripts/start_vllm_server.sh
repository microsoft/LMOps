#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../ && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="corag/CoRAG-Llama3.1-8B-MultihopQA"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if nc -z localhost 8000; then
  echo "VLLM server already running."
else
  echo "Starting VLLM server..."

  PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
  vllm serve "${MODEL_NAME_OR_PATH}" \
    --dtype auto \
    --disable-log-requests --disable-custom-all-reduce \
    --enable_chunked_prefill --max_num_batched_tokens 2048 \
    --tensor-parallel-size "${PROC_PER_NODE}" \
    --max-model-len 8192 \
    --gpu_memory_utilization 0.5 \
    --api-key token-123 > vllm_server.log 2>&1 &

  elapsed=0
  while ! nc -z localhost 8000; do
    sleep 10
    elapsed=$((elapsed + 10))
    if [ $elapsed -ge 600 ]; then
      echo "Server did not start within 10 minutes. Exiting."
      exit 1
    fi
  done
  echo "VLLM server started."
fi
