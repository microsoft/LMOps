#!/bin/bash
set -x

export NCCL_TIMEOUT=36000

# Parse command-line arguments
ROLLOUT_N=8
BATCH_SIZE=64
KL_LOSS_TYPE=full
KL_TOPK=256
ACTOR_LR=1e-6
KL_RENORM_TOPK=False
JSD_BETA=-1
JSD_FULL_TOPK=False
MAX_PROMPT_LENGTH=16384
MAX_RESPONSE_LENGTH=4096
EXP_MODEL_PATH=""
RM_PROMPT_VERSION=v2
USE_RUBRIC=4o
ITER_TEACHER_STEPS=-1
REF_MODEL_PATH=""
SAVE_FREQ=25
USE_DATASET=wildchat
MULTIDATA_RATIO=""
MULTIDATA_RM_PROMPT=""
MULTIDATA_FIXED_TEACHER=""
PPO_MAX_TOKEN=""
GPU_MEM_UTIL=0.6
SAVE_LOGPROB=False
TRAIN_TOPP=1.0
TIS_IMP_RATIO_CAP=-1
CALCULATE_LOG_PROBS=False
ROLLOUT_LOGPROBS=0
RESUME_POLICY_NAME=""
RESUME_POLICY_CKPT=""
SAVE_OPTIM=False
AUTO_RESUME=False

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --exp_model_path)
            EXP_MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --rollout_n)
            ROLLOUT_N="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --kl_loss_type)
            KL_LOSS_TYPE="$2"
            shift 2
            ;;
        --kl_topk)
            KL_TOPK="$2"
            shift 2
            ;;
        --actor_lr)
            ACTOR_LR="$2"
            shift 2
            ;;
        --kl_renorm_topk)
            KL_RENORM_TOPK="$2"
            shift 2
            ;;
        --jsd_beta)
            JSD_BETA="$2"
            shift 2
            ;;
        --jsd_full_topk)
            JSD_FULL_TOPK="$2"
            shift 2
            ;;
        --max_prompt_length)
            MAX_PROMPT_LENGTH="$2"
            shift 2
            ;;
        --max_response_length)
            MAX_RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --rm_prompt_version)
            RM_PROMPT_VERSION="$2"
            shift 2
            ;;
        --use_rubric)
            USE_RUBRIC="$2"
            shift 2
            ;;
        --iter_teacher_steps)
            ITER_TEACHER_STEPS="$2"
            shift 2
            ;;
        --ref_model_path)
            REF_MODEL_PATH="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --use_dataset)
            USE_DATASET="$2"
            shift 2
            ;;
        --multidata_ratio)
            MULTIDATA_RATIO="$2"
            shift 2
            ;;
        --multidata_rm_prompt)
            MULTIDATA_RM_PROMPT="$2"
            shift 2
            ;;
        --multidata_fixed_teacher)
            MULTIDATA_FIXED_TEACHER="$2"
            shift 2
            ;;
        --ppo_max_token)
            PPO_MAX_TOKEN="$2"
            shift 2
            ;;
        --gpu_memory_utilization)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --save_logprob)
            SAVE_LOGPROB="$2"
            shift 2
            ;;
        --train_topp)
            TRAIN_TOPP="$2"
            shift 2
            ;;
        --tis_imp_ratio_cap)
            TIS_IMP_RATIO_CAP="$2"
            CALCULATE_LOG_PROBS=True
            ROLLOUT_LOGPROBS=1
            shift 2
            ;;
        --resume_policy_name)
            RESUME_POLICY_NAME="$2"
            shift 2
            ;;
        --resume_policy_ckpt)
            RESUME_POLICY_CKPT="$2"
            shift 2
            ;;
        --save_optim)
            SAVE_OPTIM="$2"
            shift 2
            ;;
        --auto_resume)
            AUTO_RESUME="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

EXP_MODEL_PATH=${EXP_MODEL_PATH:-$MODEL_PATH}
REF_MODEL_PATH=${REF_MODEL_PATH:-$MODEL_PATH}
if [ -n "$PPO_MAX_TOKEN" ]; then
    PPO_MAX_TOKEN_LEN=$PPO_MAX_TOKEN
else
    PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
fi
PPO_MINI_BATCH_SIZE=$((BATCH_SIZE * ROLLOUT_N))
EL_DATA_ROOT=${EL_DATA_ROOT:-/tmp/el/data}
EL_CHECKPOINT_ROOT=${EL_CHECKPOINT_ROOT:-/tmp/el/checkpoints}

# Helper: resolve a single dataset name to its train parquet path
resolve_dataset_path() {
    local ds="$1"
    case "$ds" in
        wildchat)
            echo "${EL_DATA_ROOT}/wildchat-if_rubric-${USE_RUBRIC}_train.parquet"
            ;;
        tulu3)
            echo "${EL_DATA_ROOT}/tulu-3-sft-mixture-filtered.parquet"
            ;;
        *)
            echo "ERROR: unsupported dataset '$ds'" >&2
            return 1
            ;;
    esac
}

# Set dataset paths based on dataset and rubric source
# Support comma-separated multi-dataset: --use_dataset wildchat,tulu3
IFS=',' read -ra DATASET_ARRAY <<< "$USE_DATASET"
NUM_DATASETS=${#DATASET_ARRAY[@]}

if [ "$NUM_DATASETS" -gt 1 ]; then
    # Multi-dataset mode
    TRAIN_FILES=""
    for ds in "${DATASET_ARRAY[@]}"; do
        p=$(resolve_dataset_path "$ds") || exit 2
        if [ -z "$TRAIN_FILES" ]; then
            TRAIN_FILES="$p"
        else
            TRAIN_FILES="${TRAIN_FILES},${p}"
        fi
    done
    PPO_MINI_BATCH_SIZE=$((PPO_MINI_BATCH_SIZE * NUM_DATASETS))
    VAL_FILE=${EL_DATA_ROOT}/wildchat-if_rubric-${USE_RUBRIC}_val.parquet
    TRAIN_FILE_ARG="data.train_files=[${TRAIN_FILES}]"
    MULTIDATA_RATIO_ARG="data.multidata_ratio=[${MULTIDATA_RATIO}]"
    MULTIDATA_RM_PROMPT_ARG="trainer.multidata_rm_prompt=[${MULTIDATA_RM_PROMPT}]"
    if [ -n "$MULTIDATA_FIXED_TEACHER" ]; then
        MULTIDATA_FIXED_TEACHER_ARG="trainer.multidata_fixed_teacher=[${MULTIDATA_FIXED_TEACHER}]"
    else
        MULTIDATA_FIXED_TEACHER_ARG=""
    fi
else
    if [ "$USE_DATASET" != "wildchat" ]; then
        echo "ERROR: single-dataset mode only supports wildchat" >&2
        exit 2
    fi
    TRAIN_FILE=${EL_DATA_ROOT}/wildchat-if_rubric-${USE_RUBRIC}_train.parquet
    VAL_FILE=${EL_DATA_ROOT}/wildchat-if_rubric-${USE_RUBRIC}_val.parquet
    TRAIN_FILE_ARG="data.train_files=${TRAIN_FILE}"
    MULTIDATA_RATIO_ARG=""
    MULTIDATA_RM_PROMPT_ARG=""
    MULTIDATA_FIXED_TEACHER_ARG=""
fi

export HYDRA_FULL_ERROR=1
export WANDB_INIT_TIMEOUT=600
export HF_HOME=${HF_HOME:-/tmp/huggingface}
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=${WANDB_PROJECT:-el}
export WANDB_RESUME=never

if [ -n "$RESUME_POLICY_NAME" ] && [ -n "$RESUME_POLICY_CKPT" ]; then
    resume_dir="${EL_CHECKPOINT_ROOT}/${RESUME_POLICY_NAME}/global_step_${RESUME_POLICY_CKPT}/actor"
    model_path="${resume_dir}/huggingface"
    if [ -d "$model_path" ] && [ "$(find "$model_path" -maxdepth 1 -type f -print -quit 2>/dev/null)" ]; then
        echo "Huggingface directory already exists, skipping merge: $model_path"
    elif [ -d "$resume_dir" ] && [ "$(find "$resume_dir" -maxdepth 1 -name "*.pt" -type f -print -quit 2>/dev/null)" ]; then
        mkdir -p "$model_path"
        find "$resume_dir" -maxdepth 1 -type f ! -name "*.pt" -exec cp {} "$model_path" \;
        python tools/merge_model2hf.py --local_dir "$resume_dir"
    else
        echo "ERROR: checkpoint not found at ${resume_dir}. Download it before using --resume_policy_name." >&2
        exit 2
    fi
    MODEL_PATH=$model_path
    EXP_MODEL_PATH=${EXP_MODEL_PATH:-$MODEL_PATH}
    REF_MODEL_PATH=${REF_MODEL_PATH:-$MODEL_PATH}
fi

if [ "$SAVE_OPTIM" = "True" ] || [ "$SAVE_OPTIM" = "true" ]; then
    SAVE_CONTENTS="['model','extra','optimizer']"
else
    SAVE_CONTENTS="['model','extra']"
fi

LOCAL_DIR=${EL_CHECKPOINT_ROOT}/${EXP_NAME}
if [ "$AUTO_RESUME" = "True" ] || [ "$AUTO_RESUME" = "true" ]; then
    RESUME_MODE=auto
else
    RESUME_MODE=disable
fi

if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.prompt_key=prompt \
        ${TRAIN_FILE_ARG} \
        data.val_files=$VAL_FILE \
        data.train_batch_size=${BATCH_SIZE} \
        data.val_batch_size=1 \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.truncation=right \
        actor_rollout_ref.model.path=$MODEL_PATH  \
        actor_rollout_ref.model.ref_model_path=$REF_MODEL_PATH  \
        actor_rollout_ref.model.exp_model_path=$EXP_MODEL_PATH  \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
        actor_rollout_ref.actor.kl_topk=${KL_TOPK} \
        actor_rollout_ref.actor.kl_renorm_topk=${KL_RENORM_TOPK} \
        actor_rollout_ref.actor.jsd_beta=${JSD_BETA} \
        actor_rollout_ref.actor.jsd_full_topk=${JSD_FULL_TOPK} \
        actor_rollout_ref.actor.profile_kl=False \
        actor_rollout_ref.actor.save_logprob=${SAVE_LOGPROB} \
        actor_rollout_ref.actor.tis_imp_ratio_cap=${TIS_IMP_RATIO_CAP} \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.checkpoint.save_contents="${SAVE_CONTENTS}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=${TRAIN_TOPP} \
        actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL} \
        actor_rollout_ref.rollout.n=${ROLLOUT_N} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        trainer.stage=consolidate \
        trainer.use_exp_model=True \
        trainer.rm_prompt_version=${RM_PROMPT_VERSION} \
        trainer.iter_teacher_steps=${ITER_TEACHER_STEPS} \
        trainer.experience_path=no_exist.txt \
        trainer.no_think=True \
        trainer.val_before_train=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${WANDB_PROJECT} \
        trainer.experiment_name=${EXP_NAME} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.test_freq=10000000000 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=3 "${@:1}" \
        trainer.total_training_steps=300 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.enable_sleep_hack=True \
        actor_rollout_ref.rollout.calculate_log_probs=${CALCULATE_LOG_PROBS} \
        actor_rollout_ref.rollout.logprobs=${ROLLOUT_LOGPROBS} \
        trainer.default_local_dir=${LOCAL_DIR} \
        trainer.resume_mode=${RESUME_MODE} \
        ${MULTIDATA_RATIO_ARG} \
        ${MULTIDATA_RM_PROMPT_ARG} \
        ${MULTIDATA_FIXED_TEACHER_ARG}
else
    sleep infinity
fi
