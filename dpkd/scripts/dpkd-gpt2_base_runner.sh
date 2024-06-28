#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1 #${3-1}
OPTS=""

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"PATH_TO_CODE"}
DATA_BASE=PATH_TO_DATA
CKPT_NAME="gpt2-base"
CKPT="${PATH_TO_DATA}/gpt2/train/sft/${CKPT_NAME}/"
# CKPT="gpt2-large" # download automatically   
TEACHER_CKPT_NAME="xlarge-sft"
TEACHER_CKPT="${PATH_TO_DATA}/gpt2/train/sft/gpt2-xlarge/"
# data
DATA_DIR="${PATH_TO_DATA}/processed_data/dolly/pseudo-eval-new/gpt2/"
OPTS+=" --dev-num 500"
# hp
BATCH_SIZE=8
LR=0.0005
GRAD_ACC=1
EVAL_BATCH_SIZE=8
# length
MAX_LENGTH=512
# runtime

# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 4"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
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
OPTS+=" --mid-log-num -1"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type kd"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
### DPKD
OPTS+=" --epochs 30"
OPTS+=" --seqKD-DPO"
OPTS+=" --bf16"
OPTS+=" --kd-ratio 0.5"

SAVE_PATH="${BASE_PATH}/results/gpt2/train/"
OPTS+=" --save ${SAVE_PATH}"

# seed
SEED=40
OPTS+=" --seed ${SEED}"

# db
OPTS+=" --db-tmax"
OPTS+=" --save-every-epoch"

OPTS+=" --LMPT"
OPTS+=" --lm-coef 0.5"

export NCCL_DEBUG=""
export WANDB_DISABLED=True
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/main.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
