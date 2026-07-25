#!/bin/bash
set -x

export NCCL_TIMEOUT=36000

# Parse command-line arguments
ROLLOUT_N=8
BATCH_SIZE=64
ACTOR_LR=1e-6
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=4096
EXP_MODEL_PATH=""
USE_RUBRIC=4o
SAVE_FREQ=25
PPO_MAX_TOKEN=""
SAVE_LOGPROB=False
TRAIN_TOPP=1.0
TIS_IMP_RATIO_CAP=-1
CALCULATE_LOG_PROBS=False
ROLLOUT_LOGPROBS=0
SAVE_OPTIM=False
AUTO_RESUME=True
GPU_MEM_UTIL=0.6

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
        --actor_lr)
            ACTOR_LR="$2"
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
        --use_rubric)
            USE_RUBRIC="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --ppo_max_token)
            PPO_MAX_TOKEN="$2"
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
        --save_optim)
            SAVE_OPTIM="$2"
            shift 2
            ;;
        --auto_resume)
            AUTO_RESUME="$2"
            shift 2
            ;;
        --gpu_memory_utilization)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

EXP_MODEL_PATH=${EXP_MODEL_PATH:-$MODEL_PATH}
if [ -n "$PPO_MAX_TOKEN" ]; then
    PPO_MAX_TOKEN_LEN=$PPO_MAX_TOKEN
else
    PPO_MAX_TOKEN_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
fi
PPO_MINI_BATCH_SIZE=$((BATCH_SIZE * ROLLOUT_N))

EL_DATA_ROOT=${EL_DATA_ROOT:-/tmp/el/data}
EL_CHECKPOINT_ROOT=${EL_CHECKPOINT_ROOT:-/tmp/el/checkpoints}
TRAIN_FILE=${EL_DATA_ROOT}/wildchat-if_rubric-${USE_RUBRIC}_train.parquet
VAL_FILE=${EL_DATA_ROOT}/wildchat-if_rubric-${USE_RUBRIC}_val.parquet

export HYDRA_FULL_ERROR=1
export WANDB_INIT_TIMEOUT=600
export HF_HOME=${HF_HOME:-/tmp/huggingface}
export TOKENIZERS_PARALLELISM=true
export WANDB_PROJECT=${WANDB_PROJECT:-el}
export WANDB_RESUME=never

if [ "$SAVE_OPTIM" = "True" ] || [ "$SAVE_OPTIM" = "true" ]; then
    SAVE_CONTENTS="['model','extra','optimizer']"
else
    SAVE_CONTENTS="['model','extra']"
fi

if [ "$AUTO_RESUME" = "True" ] || [ "$AUTO_RESUME" = "true" ]; then
    RESUME_MODE=auto
else
    RESUME_MODE=disable
fi

if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.prompt_key=prompt \
        data.train_files=$TRAIN_FILE \
        data.val_files=$VAL_FILE \
        data.train_batch_size=${BATCH_SIZE} \
        data.val_batch_size=1 \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.truncation=right \
        actor_rollout_ref.model.path=$MODEL_PATH  \
        actor_rollout_ref.model.exp_model_path=$EXP_MODEL_PATH  \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.profile_kl=False \
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
        trainer.stage=rl \
        trainer.use_exp_model=True \
        trainer.rm_prompt_version=v1 \
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
        actor_rollout_ref.actor.save_logprob=${SAVE_LOGPROB} \
        actor_rollout_ref.actor.tis_imp_ratio_cap=${TIS_IMP_RATIO_CAP} \
        actor_rollout_ref.rollout.calculate_log_probs=${CALCULATE_LOG_PROBS} \
        actor_rollout_ref.rollout.logprobs=${ROLLOUT_LOGPROBS} \
        trainer.default_local_dir=${EL_CHECKPOINT_ROOT}/${EXP_NAME} \
        trainer.resume_mode=${RESUME_MODE}
else
    sleep infinity
fi
