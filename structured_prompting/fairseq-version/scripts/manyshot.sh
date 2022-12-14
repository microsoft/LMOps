#!/usr/bin/env bash

set -ex

# args: checkpoint_folder step

SEED=$1
TASK=$2
MODEL_PATH=$3
ARCH=$4
K=$5
CHUNK_LEN=$6
BSZ=$7
NGPU=$8
BPE_PATH=$9
ENCODER_PATH=${10}
DICT_PATH=${11}
OUTPUT_PATH=${12}


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

if [ -e $OUTPUT_PATH/train_log.txt ]; then
    if grep -q 'done training' $OUTPUT_PATH/train_log.txt; then
        echo "Training log existed"
        exit 0
    fi
fi

BSZ=$((N_CLASSES*BSZ))

python -m torch.distributed.launch --nproc_per_node=$NGPU --nnodes=1 validate.py "-" \
    --task large_icl_eval \
    --tokens-per-sample 2048  \
    --criterion large_icl_eval \
    --arch $ARCH  \
    --gpt2-vocab-bpe $BPE_PATH  \
    --gpt2-encoder-json $ENCODER_PATH \
    --batch-size 8  \
    --log-format simple  \
    --max-epoch 1 \
    --required-batch-size-multiple 1 \
    --log-interval 4 \
    --optimizer adam  \
    --warmup-updates 0  \
    --max-update 0 \
    --fp16 \
    --eval-data $TASK \
    --checkpoint-activations \
    --fp16-init-scale 4 \
    --no-save \
    --fp16-scale-window 256 \
    --seed $SEED \
    --lr 0.0  \
    --k $K \
    --truncate \
    --chunk-len $CHUNK_LEN \
    --batch-size-valid $BSZ \
    --ddp-backend=no_c10d \
    --distributed-no-spawn \
    --valid-subset demons,valid \
    --gpt-dict $DICT_PATH \
    --gpt-model-path $MODEL_PATH 
