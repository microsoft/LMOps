#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="intfloat/e5-base-v2"
if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/kd_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/tasks/"
fi

deepspeed src/train_biencoder.py --deepspeed ds_config.json \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_checkpointing True \
    --kd_cont_loss_weight 0.2 \
    --l2_normalize True --t 0.01 \
    --pool_type avg \
    --seed 123 \
    --do_train \
    --do_kd_biencoder \
    --fp16 \
    --train_file "${DATA_DIR}/kd_bm25_train.jsonl.gz,${DATA_DIR}/kd_it2_train.jsonl.gz" \
    --max_len 256 \
    --train_n_passages 4 \
    --topk_as_positive 3 --bottomk_as_negative 16 \
    --dataloader_num_workers 1 \
    --max_steps 6000 \
    --learning_rate 3e-5 \
    --warmup_steps 400 \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 10 \
    --save_strategy steps \
    --save_steps 2000 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@"
