#!/bin/bash

MODEL=yahma/llama-7b-hf

datadir=$1
NO_DISCRIMINATE=False
LENPEN=1.0
savedir=./checkpoints/tuna_p
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=1
SAVE_STEPS=0.99
LR=$2
MLE_WEIGHT=1.0
MARGIN=0.1
BETA1=0.9
BETA2=0.999
WARMUP_STEPS=2
REMOVE_UNUSED_COLUMNS=False


deepspeed train_tuna.py \
    --model_name_or_path ${MODEL} \
    --data_path $datadir \
    --no_discriminate $NO_DISCRIMINATE \
    --lenpen $LENPEN \
    --output_dir $savedir \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps $SAVE_STEPS \
    --save_total_limit 1 \
    --learning_rate $LR \
    --mle_weight $MLE_WEIGHT \
    --margin $MARGIN \
    --adam_beta1 $BETA1 \
    --adam_beta2 $BETA2 \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --log_level debug \
    --remove_unused_columns $REMOVE_UNUSED_COLUMNS \
    --deepspeed ./configs/deepspeed_config.json \
    --fp16 True 2>&1 | tee training.log