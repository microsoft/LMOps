#!/bin/bash
set -x

export NCCL_TIMEOUT=36000



# Parse command line arguments
ROLLOUT_N=1
KL_LOSS_TYPE=full
KL_TOPK=256
ACTOR_LR=1e-6
KL_RENORM_TOPK=False
MAX_RESPONSE_LENGTH=1024
REF_MODEL_PATH=""
TOTAL_TRAINING_STEPS=50
SAVE_FREQ=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --ref_model_path)
            REF_MODEL_PATH="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --exp_path)
            EXP_PATH="$2"
            shift 2
            ;;
        --experience_max_length)
            EXPERIENCE_MAX_LENGTH="$2"
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
        --off_policy_save_dir)
            OFF_POLICY_SAVE_DIR="$2"
            shift 2
            ;;
        --max_response_length)
            MAX_RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --system_prompt_type)
            SYSTEM_PROMPT_TYPE="$2"
            shift 2
            ;;
        --total_training_steps)
            TOTAL_TRAINING_STEPS="$2"
            shift 2
            ;;
        --save_freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

REF_MODEL_PATH=${REF_MODEL_PATH:-$MODEL_PATH}

MAX_PROMPT_LENGTH=$((EXPERIENCE_MAX_LENGTH + 1024))
PPO_MAX_TOKEN_LEN=20480

export TOKENIZERS_PARALLELISM=true
export WANDB_INIT_TIMEOUT=600
export WANDB_RESUME=never
export HYDRA_FULL_ERROR=1

wandb login ${WANDB_API_KEY}


if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
        data.prompt_key=content \
        data.train_files=/tmp/sys_${SYSTEM_PROMPT_TYPE}_train.parquet \
        data.val_files=/tmp/sys_${SYSTEM_PROMPT_TYPE}_test.parquet \
        data.train_batch_size=128 \
        data.val_batch_size=1 \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.truncation=right \
        actor_rollout_ref.model.path=$MODEL_PATH  \
        actor_rollout_ref.model.ref_model_path=$REF_MODEL_PATH  \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.rollout.max_num_batched_tokens=${PPO_MAX_TOKEN_LEN} \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE} \
        actor_rollout_ref.actor.kl_topk=${KL_TOPK} \
        actor_rollout_ref.actor.kl_renorm_topk=${KL_RENORM_TOPK} \
        actor_rollout_ref.actor.profile_kl=False \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.checkpoint.save_contents="['model','extra']" \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=${ROLLOUT_N} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.stage=consolidate \
        trainer.train_system_prompt=True \
        trainer.system_prompt_type=${SYSTEM_PROMPT_TYPE} \
        trainer.system_prompt_version=v1 \
        trainer.experience_path=${EXP_PATH} \
        trainer.experience_max_length=${EXPERIENCE_MAX_LENGTH} \
        trainer.val_before_train=False \
        trainer.on_policy_merge=False \
        trainer.generate_off_policy=False \
        trainer.off_policy_save_dir=${OFF_POLICY_SAVE_DIR} \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name=${WANDB_PROJECT} \
        trainer.experiment_name=${EXP_NAME} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=${SAVE_FREQ} \
        trainer.test_freq=10000000000 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=10000000000 "${@:1}" \
        trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.enable_sleep_hack=True \
        trainer.default_local_dir=/tmp/${EXP_NAME}
else
    sleep infinity
fi