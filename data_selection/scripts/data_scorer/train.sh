#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-8}
NNODES=${4-2}
HOSTFILE=${5-hostfile_8V100_0_1}

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT \
                  --hostfile $BASE_PATH/configs/hostfiles/$HOSTFILE"

# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT_NAME="fairseq/125M"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
# data
DATA_DIR="${BASE_PATH}/processed_data/data_scorer_train/cc-sgd100-160M-10k-lima-163840"
DATA_NAME="cc-sgd100-160M-10k-lima-163840"

# hp
BATCH_SIZE=16
LR=0.0001
GRAD_ACC=2
EVAL_BATCH_SIZE=64
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/data_scorer"
# seed
SEED=10


OPTS=""
# type
OPTS+=" --type data_scorer"
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --model-type fairseq"
OPTS+=" --data-scorer-encoding mean"
OPTS+=" --data-scorer-bias"
OPTS+=" --data-scorer-head-type linear"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --train-num 163840"
OPTS+=" --dev-num 16384"
OPTS+=" --data-name ${DATA_NAME}"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 10"
OPTS+=" --scheduler-name cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 5"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 1"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# wandb
OPTS+=" --wandb-mode disabled"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
