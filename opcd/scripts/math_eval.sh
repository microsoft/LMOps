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
        *)
            break
            ;;
    esac
done

MAX_PROMPT_LENGTH=1024
PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH))

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
        data.train_files=/tmp/dapo_train.parquet \
        data.val_files=/tmp/dapo_test.parquet \
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
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        trainer.stage=extract \
        trainer.eval_wo_experience=True \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.val_samples_limit=20 \
        trainer.held_out_size=1000 \
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