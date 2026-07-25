#!/bin/bash
set -x

export NCCL_TIMEOUT=36000
# Parse command-line arguments
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
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --use_bsl)
            USE_BSL="$2"
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
        --use_dataset)
            USE_DATASET="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

USE_BSL=${USE_BSL:-false}
EVAL_PREPEND_EXPERIENCE=${EVAL_PREPEND_EXPERIENCE:-false}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-16384}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
USE_DATASET=${USE_DATASET:-wildchat_4o}
EXP_MODEL_PATH=${EXP_MODEL_PATH:-gpt4o}
EL_DATA_ROOT=${EL_DATA_ROOT:-/tmp/el/data}
EL_CHECKPOINT_ROOT=${EL_CHECKPOINT_ROOT:-/tmp/el/checkpoints}
EL_RESULT_ROOT=${EL_RESULT_ROOT:-/tmp/el/results}

# Check if -train suffix: strip it, use train file as val, limit samples
USE_TRAIN_AS_VAL=false
HELD_OUT_SIZE=500
if [[ "$USE_DATASET" == *"-train" ]]; then
    USE_TRAIN_AS_VAL=true
    HELD_OUT_SIZE=250
    USE_DATASET="${USE_DATASET%-train}"
fi

if [ "$USE_DATASET" != "wildchat" ] && [ "$USE_DATASET" != "wildchat_4o" ]; then
    echo "ERROR: eval only supports wildchat_4o" >&2
    exit 2
fi

TRAIN_FILE=${EL_DATA_ROOT}/wildchat-if_rubric-4o_train.parquet
VAL_FILE=${EL_DATA_ROOT}/wildchat-if_rubric-4o_val.parquet

if [ "$USE_TRAIN_AS_VAL" = "true" ]; then
    VAL_FILE=$TRAIN_FILE
fi

EVAL_SUFFIX=evaluation
if [ "$USE_TRAIN_AS_VAL" = "true" ]; then
    EVAL_SUFFIX="${EVAL_SUFFIX}_trainset"
fi

PPO_MAX_TOKEN_LEN_PER_GPU=20480

export HYDRA_FULL_ERROR=1
export HF_HOME=${HF_HOME:-/tmp/huggingface}
export TOKENIZERS_PARALLELISM=true

# merge checkpoint
if [ "$USE_BSL" = "true" ]; then
    echo "Skipping model merge as USE_BSL is true"
    final_model_path=$MODEL_PATH
else
    resume_dir="${EL_CHECKPOINT_ROOT}/${EXP_NAME}/global_step_${CKPT}/actor"
    model_path="${resume_dir}/huggingface"
    if [ -d "$model_path" ] && [ "$(find "$model_path" -maxdepth 1 -type f -print -quit 2>/dev/null)" ]; then
        echo "Huggingface directory already exists and has files, skipping model preparation: $model_path"
    elif [ -d "$resume_dir" ] && [ "$(find "$resume_dir" -maxdepth 1 -name "*.pt" -type f -print -quit 2>/dev/null)" ]; then
        mkdir -p "$model_path"
        find "$resume_dir" -maxdepth 1 -type f ! -name "*.pt" -exec cp {} "$model_path" \;
        python tools/merge_model2hf.py --local_dir "$resume_dir"
    else
        echo "ERROR: checkpoint not found at ${resume_dir}. Download it before evaluation." >&2
        exit 2
    fi
    final_model_path=$model_path
fi

if [ "${OMPI_COMM_WORLD_RANK:-0}" -eq 0 ]; then
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.prompt_key=prompt \
        data.train_files=$TRAIN_FILE \
        data.val_files=$VAL_FILE \
        data.train_batch_size=64 \
        data.val_batch_size=1 \
        data.max_prompt_length=$MAX_PROMPT_LENGTH \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        data.truncation=right \
        actor_rollout_ref.model.path=$final_model_path  \
        actor_rollout_ref.model.exp_model_path=$EXP_MODEL_PATH  \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
        actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
        actor_rollout_ref.rollout.val_kwargs.top_k=20 \
        trainer.stage=consolidate \
        trainer.use_exp_model=True \
        trainer.rm_prompt_version=v1 \
        trainer.eval_wo_experience=True \
        trainer.eval_prepend_experience=False \
        trainer.experience_path=no_exist.txt \
        trainer.no_think=True \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.held_out_size=${HELD_OUT_SIZE} \
        trainer.held_out_rollout=4 \
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
        trainer.validation_data_dir=${EL_RESULT_ROOT}/${EXP_NAME}/global_step_${CKPT}/${EVAL_SUFFIX}
else
    sleep infinity
fi
