#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_ADDR=localhost
MASTER_PORT=${2-2030}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-8}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# type
TYPE="eval_lm"
# model
CKPT_NAME="1B_bsl"
CKPT="${BASE_PATH}/results/pretrain/1B_bsl"
# data
DATA_NAME="dclm"
# hp
EVAL_BATCH_SIZE=64
# runtime
SAVE_PATH="${BASE_PATH}/results/${TYPE}"
# seed
SEED=10
# wandb
WANDB_NAME="1B_bsl"


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
# data
OPTS+=" --data-dir ${BASE_PATH}/processed_data/dclm/mistral-1025"
OPTS+=" --data-name ${DATA_NAME}"
OPTS+=" --data-split test"
OPTS+=" --bin-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
# runtime
OPTS+=" --eval-interval 5000"
OPTS+=" --eval-start-ckpt 5000"
OPTS+=" --eval-end-ckpt 100000"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --wandb-group eval_harness"
OPTS+=" --wandb-name ${WANDB_NAME}"
OPTS+=" --wandb-mode disabled"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"


export NCCL_DEBUG=""
# export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/eval_offline.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
