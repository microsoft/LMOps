#! /bin/bash

BASE_PATH=${1-"/home/MiniLLM"}
MASTER_PORT=${2-2030}
GPUS_PER_NODE=${3-8}
NNODES=${4-2}
HOSTFILE=${5-node_0_1}

DISTRIBUTED_ARGS="--num_gpus $GPUS_PER_NODE \
                  --num_nodes $NNODES \
                  --hostfile $BASE_PATH/configs/hostfiles/$HOSTFILE \
                  --master_port $MASTER_PORT"

# model
CKPT_NAME="mistral-7B"
CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
MP_SIZE=2
# data
DATA_DIR="${BASE_PATH}/processed_data/dolly/full/mistral/"
# hp
BATCH_SIZE=4
LR=0.00001
GRAD_ACC=1
EVAL_BATCH_SIZE=8
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/mistral/train/sft"
# seed
SEED=10
SEED_ORDER=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --model-type mistral"
OPTS+=" --gradient-checkpointing"
OPTS+=" --model-parallel"
OPTS+=" --model-parallel-size ${MP_SIZE}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --dev-num 500"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 10"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num 1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-order ${SEED_ORDER}"
# deepspeed
OPTS+=" --deepspeed"
# OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2_offload.json"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config_zero2.json"
# type
OPTS+=" --type lm"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
