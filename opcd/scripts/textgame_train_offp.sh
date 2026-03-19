#!/bin/bash
set -x

export NCCL_TIMEOUT=36000

# Parse command line arguments
KL_LOSS_TYPE=full
KL_TOPK=256
ACTOR_LR=1e-6
KL_RENORM_TOPK=False
MAX_RESPONSE_LENGTH=1024
EXPERIENCE_MAX_LENGTH=8192

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
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
        --nnodes)
            NNODES="$2"
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
        --experience_max_length)
            EXPERIENCE_MAX_LENGTH="$2"
            shift 2
            ;;
        --textgame_name)
            TEXTGAME_NAME="$2"
            shift 2
            ;;
        --textgame_max_steps)
            TEXTGAME_MAX_STEPS="$2"
            shift 2
            ;;
        --textgame_no_think)
            TEXTGAME_NO_THINK="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

MAX_PROMPT_LENGTH=$((EXPERIENCE_MAX_LENGTH + 512 * TEXTGAME_MAX_STEPS))

TEXTGAME_MAX_PROMPT_LENGTH=$MAX_PROMPT_LENGTH

PPO_MAX_TOKEN_LEN=20480

export TOKENIZERS_PARALLELISM=true
export WANDB_INIT_TIMEOUT=600
export WANDB_RESUME=never
export HYDRA_FULL_ERROR=1

wandb login ${WANDB_API_KEY}



if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
        data.prompt_key=content \
        data.train_batch_size=64 \
        data.val_batch_size=1 \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.truncation=right \
        actor_rollout_ref.model.path=$MODEL_PATH  \
        actor_rollout_ref.actor.optim.lr=${ACTOR_LR} \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128000 \
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
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        trainer.stage=consolidate \
        trainer.setting='textgame' \
        trainer.skip_reward=True \
        trainer.textgame_env_id=${TEXTGAME_NAME} \
        trainer.textgame_max_steps=${TEXTGAME_MAX_STEPS} \
        trainer.textgame_wfeedback=True \
        trainer.textgame_keep_reasoning=True \
        trainer.textgame_max_prompt_length=$TEXTGAME_MAX_PROMPT_LENGTH \
        trainer.textgame_max_response_length=${MAX_RESPONSE_LENGTH} \
        trainer.experience_max_length=${EXPERIENCE_MAX_LENGTH} \
        trainer.textgame_no_think=${TEXTGAME_NO_THINK} \
        trainer.textgame_total_steps=50 \
        trainer.experience_path=${EXP_PATH} \
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
        trainer.save_freq=2 \
        trainer.test_freq=10000000000 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=10000000000 "${@:1}" \
        trainer.total_training_steps=50 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.enable_sleep_hack=True \
        trainer.default_local_dir=/tmp/${EXP_NAME}
else
    sleep infinity
fi