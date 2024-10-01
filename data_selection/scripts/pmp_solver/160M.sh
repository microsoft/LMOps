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


# hp
LR=0.008
BATCH_SIZE=8
GRAD_ACC=2
# runtime
SAVE_PATH="${BASE_PATH}/results/pmp_solver/"
# seed
SEED=10
SEED_DATA=20


OPTS=""
OPTS+=" --type pmp_solver"
# model
OPTS+=" --model-type mistral"
OPTS+=" --model-path ${BASE_PATH}/results/pretrain/cc/mistral_160M/t100K-w2K-bs8-lr0.0006cosine6e-05-G4-N16-NN2-scr/10000"
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --ckpt-name 160M-10k"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --attn-impl eager"
OPTS+=" --fp32"
# data
OPTS+=" --max-state 10"
OPTS+=" --proxy-num 163840"
OPTS+=" --data-name cc-lima"
OPTS+=" --data-dir ${BASE_PATH}/processed_data/pretrain/cc/mistral-1025"
OPTS+=" --dev-data-dir ${BASE_PATH}/processed_data/lima/mistral-1025/dev"
OPTS+=" --proxy-data-dir ${BASE_PATH}/processed_data/proxy/cc-mistral-1025/163840"
OPTS+=" --bin-data"
OPTS+=" --no-shuffle"
OPTS+=" --dataset-type lm"
# hp
OPTS+=" --total-iters 100"
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --grad-batch-size 2"
OPTS+=" --proxy-batch-size 8"
OPTS+=" --eval-batch-size 4"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --clip-grad -1"
OPTS+=" --max-length 1024"
OPTS+=" --num-workers 2"
OPTS+=" --weight-decay 0.0"
OPTS+=" --optimizer-name sgd"
OPTS+=" --scheduler-name constant"
OPTS+=" --warmup-iters 0"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-interval 1"
OPTS+=" --eval-interval 10"
OPTS+=" --wandb-group pmp_solver"
OPTS+=" --compute-ct-interval 10"
# seed
OPTS+=" --seed ${SEED}"
OPTS+=" --seed-data ${SEED_DATA}"
# wandb
OPTS+=" --wandb-mode disabled"


export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export OMP_NUM_THREADS=16
CMD="deepspeed ${DISTRIBUTED_ARGS} ${BASE_PATH}/train.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}