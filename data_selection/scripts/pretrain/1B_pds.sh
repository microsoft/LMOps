#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-8}
NNODES=${4-2}
HOSTFILE=${5-hostfile_8V100_0_1}

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --master_port $MASTER_PORT
                  --hostfile $BASE_PATH/configs/hostfiles/$HOSTFILE"

# type
TYPE="pretrain"
# model
CKPT_NAME="mistral/1B"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
# data
DATA_DIR="${BASE_PATH}/processed_data/pretrain/cc-sgd100-160M-10k-lima-t0.1-r0.4"
DEV_LM_DATA_DIR="${BASE_PATH}/processed_data/pretrain_eval/cc-mistral-1025/dev-100K"
# hp
BATCH_SIZE=8
LR=0.00025
LR_MIN=0.000025
GRAD_ACC=4
EVAL_BATCH_SIZE=64
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10
# NAMES
WANDB_NAME="1B-sgd100-160M-10k-lima-t0.1-r0.4"
DATA_NAME="cc-sgd100-160M-10k-lima-t0.1-r0.4"


OPTS=""
# type
OPTS+=" --type ${TYPE}"
# model
OPTS+=" --model-type mistral"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
# OPTS+=" --gradient-checkpointing"
OPTS+=" --from-scratch"
OPTS+=" --attn-impl eager"
OPTS+=" --xops-attn"
# OPTS+=" --torch-compile reduce-overhead"
# data
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --dev-data-dir ${DEV_LM_DATA_DIR}"
OPTS+=" --num-workers 8"
OPTS+=" --dev-num 16384"
OPTS+=" --bin-data"
OPTS+=" --no-shuffle"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --lr-min ${LR_MIN}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 2000"
OPTS+=" --scheduler-name cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --adam-beta 0.9"
OPTS+=" --adam-beta2 0.98"
OPTS+=" --adam-eps 1e-6"
OPTS+=" --total-iters 100000"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --save-interval 5000"
OPTS+=" --eval-interval 1000"
OPTS+=" --log-interval 10"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --wandb-group pretrain"
OPTS+=" --wandb-name ${WANDB_NAME}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
