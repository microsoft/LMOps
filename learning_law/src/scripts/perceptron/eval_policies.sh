#! /bin/bash

BASE_PATH=${1-"/home/LearningLaw"}
MASTER_ADDR=localhost
MASTER_PORT=${2-2030}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-2}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"


# hp
LR=0.1
BATCH_SIZE=512
# runtime
SAVE_PATH="${BASE_PATH}/results/"
# seed
SEED=10
SEED_DATA=20


OPTS=""
# model
OPTS+=" --model-type perceptron"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-name 128"
# data
OPTS+=" --train-num 4096"
OPTS+=" --dev-num 512"
OPTS+=" --test-num 512"
OPTS+=" --data-names linear"
OPTS+=" --data-dir ${BASE_PATH}/data/linear/"
OPTS+=" --load-cache 1"
OPTS+=" --load-gamma ${BASE_PATH}/results/perceptron/128/bs512-lr0.1-tn4096-dn512-e2000/opt-0.001-0/10-20-7"
OPTS+=" --policy-name 0.001"
OPTS+=" --eval-gamma-epochs ."
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size 64"
OPTS+=" --grad-batch-size 512"
OPTS+=" --epochs 2000"
OPTS+=" --log-interval 10"
OPTS+=" --clip-grad -1"
OPTS+=" --max-length 64"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --eval-opt-gamma"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-data ${SEED_DATA}"


export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/src/main.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
