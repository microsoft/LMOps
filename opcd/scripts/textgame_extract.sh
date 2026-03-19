#!/bin/bash
set -x

export NCCL_TIMEOUT=36000
# Parse command line arguments
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
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --prompt_version)
            PROMPT_VERSION="$2"
            shift 2
            ;;
        --exp_sel_with_prev)
            EXP_SEL_WITH_PREV="$2"
            shift 2
            ;;
        --val_samples_limit)
            VAL_SAMPLES_LIMIT="$2"
            shift 2
            ;;
        --max_response_length)
            MAX_RESPONSE_LENGTH="$2"
            shift 2
            ;;
        --textgame_name)
            TEXTGAME_NAME="$2"
            shift 2
            ;;
        --textgame_max_response)
            TEXTGAME_MAX_RESPONSE="$2"
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
        --exp_model_path)
            EXP_MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

EXP_MODEL_PATH=${EXP_MODEL_PATH:-$MODEL_PATH}

if [ "${EXP_SEL_WITH_PREV,,}" = "false" ]; then
    MAX_PROMPT_LENGTH=$(((TEXTGAME_MAX_RESPONSE + 512) * TEXTGAME_MAX_STEPS + 1024))
else
    MAX_PROMPT_LENGTH=$((MAX_RESPONSE_LENGTH + (TEXTGAME_MAX_RESPONSE + 512) * TEXTGAME_MAX_STEPS + 1024))
fi

TEXTGAME_MAX_PROMPT_LENGTH=$((MAX_RESPONSE_LENGTH + 512 * TEXTGAME_MAX_STEPS))
if [ $TEXTGAME_MAX_PROMPT_LENGTH -gt $MAX_PROMPT_LENGTH ]; then
    MAX_PROMPT_LENGTH=$TEXTGAME_MAX_PROMPT_LENGTH
fi

if [ $((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) -gt 20480 ]; then
    PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH))
else
    PPO_MAX_TOKEN_LEN_PER_GPU=20480
fi


export TOKENIZERS_PARALLELISM=true
export WANDB_INIT_TIMEOUT=600
export WANDB_RESUME=never
export HYDRA_FULL_ERROR=1

wandb login ${WANDB_API_KEY}

if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
        data.prompt_key=content \
        data.val_batch_size=1 \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.truncation=right \
        actor_rollout_ref.model.path=$MODEL_PATH  \
        actor_rollout_ref.model.exp_model_path=$EXP_MODEL_PATH  \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        trainer.stage=extract \
        trainer.setting='textgame' \
        trainer.textgame_env_id=${TEXTGAME_NAME} \
        trainer.textgame_max_steps=${TEXTGAME_MAX_STEPS} \
        trainer.textgame_wfeedback=True \
        trainer.textgame_keep_reasoning=True \
        trainer.textgame_max_prompt_length=$TEXTGAME_MAX_PROMPT_LENGTH \
        trainer.textgame_max_response_length=$TEXTGAME_MAX_RESPONSE \
        trainer.textgame_no_think=${TEXTGAME_NO_THINK} \
        trainer.textgame_total_steps=20000 \
        trainer.prompt_version=${PROMPT_VERSION} \
        trainer.exp_sel_with_prev=${EXP_SEL_WITH_PREV} \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.val_samples_limit=${VAL_SAMPLES_LIMIT} \
        trainer.held_out_size=128 \
        trainer.held_out_rollout=1 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name="extract" \
        trainer.experiment_name="extract" \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=10000000000 \
        trainer.test_freq=10000000000 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=10000000000 "${@:1}" \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.enable_sleep_hack=True \
        +actor_rollout_ref.rollout.seed=${CKPT} \
        trainer.validation_data_dir=/tmp/${EXP_NAME}/global_step_${CKPT}/extract_${VAL_SAMPLES_LIMIT}_samples
else
    sleep infinity
fi