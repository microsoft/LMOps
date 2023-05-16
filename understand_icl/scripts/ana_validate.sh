#!/usr/bin/env bash

set -ex

SEED=$1
TASK=$2
MODEL_PATH=$3
ARCH=$4
K=$5
BSZ=$6
NGPU=$7
BPE_PATH=$8
ENCODER_PATH=$9
DICT_PATH=${10}
OUTPUT_PATH=${11}

ana_attn=${12}
ana_rlt_dir=${13}
ana_setting=${14}
perm_id=${15}

N_CLASSES=2
if [ "$TASK" = "agnews" ]
then
N_CLASSES=4
fi
if [ "$TASK" = "trec" ]
then
N_CLASSES=6
fi
if [ "$TASK" = "sst5" ]
then
N_CLASSES=5
fi
if [ "$TASK" = "dbpedia" ]
then
N_CLASSES=14
fi
if [ "$TASK" = "cb" ]
then
N_CLASSES=3
fi

if [ "$TASK" = "arce" ]
then
N_CLASSES=4
fi
if [ "$TASK" = "arcc" ]
then
N_CLASSES=4
fi
if [ "$TASK" = "obqa" ]
then
N_CLASSES=4
fi
if [ "$TASK" = "hellaswag" ]
then
N_CLASSES=4
fi
if [ "$TASK" = "storycloze" ]
then
N_CLASSES=4
fi
if [ "$TASK" = "raceh" ]
then
N_CLASSES=4
fi
if [ "$TASK" = "racem" ]
then
N_CLASSES=4
fi

if [ "$TASK" = "nq" ]
then
N_CLASSES=1
BSZ=1
extra_args="--is-generation"
fi
if [ "$TASK" = "webqs" ]
then
N_CLASSES=1
BSZ=1
extra_args="--is-generation"
fi
if [ "$TASK" = "triviaqa" ]
then
N_CLASSES=1
BSZ=1
extra_args="--is-generation"
fi
if [ "$TASK" = "sq2" ]
then
N_CLASSES=1
BSZ=1
extra_args="--is-generation"
fi
if [ "$TASK" = "sq" ]
then
N_CLASSES=1
BSZ=1
extra_args="--is-generation"
fi

OUTPUT_PATH=$OUTPUT_PATH
mkdir -p $OUTPUT_PATH
echo $OUTPUT_PATH

BSZ=$((N_CLASSES*BSZ))

python validate.py "-" \
    --task fs_eval \
    --tokens-per-sample 2048  \
    --criterion fs_eval \
    --arch $ARCH  \
    --gpt2-vocab-bpe $BPE_PATH  \
    --gpt2-encoder-json $ENCODER_PATH \
    --log-format simple  \
    --max-epoch 1 \
    --required-batch-size-multiple 1 \
    --log-interval 1 \
    --warmup-updates 0 \
    --optimizer sgd \
    --max-update 0 \
    --fp16 \
    --eval-data $TASK \
    --fp16-init-scale 4 \
    --checkpoint-activations \
    --fp16-scale-window 256 \
    --seed $SEED \
    --reset-dataloader \
    --no-save \
    --k $K \
    --batch-size $BSZ \
    --batch-size-valid $BSZ \
    --ddp-backend=no_c10d \
    --distributed-no-spawn \
    --gpt-dict $DICT_PATH \
    --gpt-model-path $MODEL_PATH \
    --ana-attn $ana_attn \
    --ana-rlt-dir $ana_rlt_dir \
    --ana-setting $ana_setting \
    --permut-index $perm_id |& tee $OUTPUT_PATH/train_log_$ana_setting.txt
