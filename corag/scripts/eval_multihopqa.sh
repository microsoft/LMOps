#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd ../ && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/tmp/"
fi

MAX_PATH_LENGTH=6
for TASK in 2wikimultihopqa bamboogle hotpotqa musique; do
    SPLIT="validation"
    if [ "${TASK}" == "bamboogle" ]; then
        SPLIT="test"
    fi

    PYTHONPATH=src/ torchrun --nproc_per_node 1 src/inference/run_inference.py \
        --eval_task "${TASK}" \
        --eval_split "${SPLIT}" \
        --max_path_length "${MAX_PATH_LENGTH}" \
        --output_dir "${OUTPUT_DIR}/${MAX_PATH_LENGTH}" \
        --do_eval \
        --num_threads 32 \
        --overwrite_output_dir \
        --disable_tqdm True \
        --report_to none "$@"
done

echo "Done"
