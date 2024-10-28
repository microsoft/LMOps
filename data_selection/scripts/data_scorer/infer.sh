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


# model
BASE_PATH=${1-"/home/MiniLLM"}
CKPT="${BASE_PATH}/results/data_scorer/cc-sgd100-160M-10k-lima-163840/fairseq_125M/e5-w10-bs16-lr0.0001cosine1e-07-G2-N16-NN2/mean-bias-linear/best"
CKPT_NAME="cc-sgd100-160M-10k-lima"
# data
DATA_DIR="${BASE_PATH}/processed_data/data_scorer_infer/cc/mistral-fairseq-1024"
# hp
BATCH_SIZE=128
# length
MAX_LENGTH=1024
# runtime
SAVE_PATH="${BASE_PATH}/results/data_scorer_infer/"
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
OPTS+=" --model-type fairseq"
OPTS+=" --attn-impl eager"
OPTS+=" --xops-attn"
OPTS+=" --torch-compile reduce-overhead"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
OPTS+=" --infer-num 160000000"
OPTS+=" --data-name cc"
# hp
OPTS+=" --eval-batch-size ${BATCH_SIZE}"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
# runtime
OPTS+=" --do-infer"
OPTS+=" --log-interval 10"
OPTS+=" --save-interval 2500"
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
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/infer.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
