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
        --use_bsl)
            USE_BSL="$2"
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

TEXTGAME_MAX_PROMPT_LENGTH=$((512 * TEXTGAME_MAX_STEPS))
MAX_PROMPT_LENGTH=$TEXTGAME_MAX_PROMPT_LENGTH

PPO_MAX_TOKEN_LEN_PER_GPU=20480

export TOKENIZERS_PARALLELISM=true
export WANDB_INIT_TIMEOUT=600
export WANDB_RESUME=never
export HYDRA_FULL_ERROR=1

wandb login ${WANDB_API_KEY}

# merge checkpoint
if [ "$USE_BSL" = "true" ]; then
    echo "Skipping model merge as USE_BSL is true"
    final_model_path=$MODEL_PATH
else
    model_path="/tmp/${EXP_NAME}/global_step_${CKPT}/actor/huggingface"
    mkdir -p /tmp/${EXP_NAME}/global_step_${CKPT}/actor/huggingface/
    find /tmp/${EXP_NAME}/global_step_${CKPT}/actor/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} /tmp/${EXP_NAME}/global_step_${CKPT}/actor/huggingface/ \;
    python tools/merge_model2hf.py --local_dir /tmp/${EXP_NAME}/global_step_${CKPT}/actor
    final_model_path=$model_path
fi

if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
        data.prompt_key=content \
        data.val_batch_size=1 \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.truncation=right \
        actor_rollout_ref.model.path=$final_model_path  \
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
        trainer.eval_wo_experience=True \
        trainer.setting='textgame' \
        trainer.textgame_env_id=${TEXTGAME_NAME} \
        trainer.textgame_max_steps=${TEXTGAME_MAX_STEPS} \
        trainer.textgame_wfeedback=True \
        trainer.textgame_keep_reasoning=True \
        trainer.textgame_max_prompt_length=$TEXTGAME_MAX_PROMPT_LENGTH \
        trainer.textgame_max_response_length=$MAX_RESPONSE_LENGTH \
        trainer.textgame_no_think=${TEXTGAME_NO_THINK} \
        trainer.textgame_total_steps=20000 \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.val_samples_limit=20 \
        trainer.held_out_size=128 \
        trainer.held_out_rollout=1 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name="eval" \
        trainer.experiment_name="eval" \
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
        trainer.validation_data_dir=/tmp/${EXP_NAME}/global_step_${CKPT}/evaluation
else
    sleep infinity
fi