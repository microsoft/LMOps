#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/reward_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/tasks/"
fi

# For electra-large, learning rate > 1e-5 will lead to instability empirically
deepspeed src/train_cross_encoder.py --deepspeed ds_config.json \
    --model_name_or_path google/electra-base-discriminator \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --do_train \
    --fp16 \
    --seed 987 \
    --train_file "${DATA_DIR}/llama-7b_bm25_train.jsonl.gz" \
    --train_n_passages 8 \
    --topk_as_positive 3 --bottomk_as_negative 16 \
    --dataloader_num_workers 1 \
    --max_steps 3000 \
    --learning_rate 1e-5 \
    --warmup_steps 400 \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 5 \
    --save_strategy steps \
    --save_steps 1000 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"
