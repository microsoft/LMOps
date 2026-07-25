#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-${EL_DATASET_REPO:-ytz20/el-data}}"
DATA_DIR="${2:-${EL_DATA_ROOT:-/tmp/el/data}}"

if ! command -v hf >/dev/null 2>&1; then
    echo "ERROR: Hugging Face CLI not found. Install it with: pip install -U huggingface_hub" >&2
    exit 127
fi

FILES=(
    wildchat-if_rubric-4o_train.parquet
    wildchat-if_rubric-4o_val.parquet
    tulu-3-sft-mixture-filtered.parquet
    alpacaeval2/alpaca_eval_gpt4_baseline.json
    wildbench/v2.json
    arena_hard_v2/prompts.json
    creativewritingv3/creative_writing_prompts_v3.json
)

mkdir -p "$DATA_DIR"

hf download "$REPO_ID" "${FILES[@]}" \
    --repo-type dataset \
    --local-dir "$DATA_DIR"

for file in "${FILES[@]}"; do
    if [ ! -f "$DATA_DIR/$file" ]; then
        echo "ERROR: missing downloaded file: $DATA_DIR/$file" >&2
        exit 1
    fi
done

echo "All EL datasets are available under $DATA_DIR"
