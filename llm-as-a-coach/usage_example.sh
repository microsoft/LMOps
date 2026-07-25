#!/usr/bin/env bash
set -euo pipefail

list_experiments() {
    printf '%s\n' \
        qwen-el-self \
        qwen-el-self-iter \
        qwen-el-gpt4o \
        olmo-el-self \
        olmo-el-gpt4o \
        qwen-rl-self \
        qwen-rl-gpt4o \
        olmo-rl-self \
        olmo-rl-gpt4o
}

usage() {
    cat <<'EOF'
Usage:
  bash usage_example.sh list
  bash usage_example.sh train <experiment>
  bash usage_example.sh eval <experiment> <checkpoint-step>
  bash usage_example.sh eval_fuzzy <experiment> <checkpoint-step>
  bash usage_example.sh eval_endtask <experiment> <checkpoint-step>
EOF
}

set_experiment_metadata() {
    case "$1" in
        qwen-el-self)
            MODEL_PATH=/tmp/Qwen3-8B
            EXP_NAME=wildchat-el-q3-8b-r8b
            ;;
        qwen-el-self-iter)
            MODEL_PATH=/tmp/Qwen3-8B
            EXP_NAME=wildchat-el-q3-8b-r8b-itert30-mopd025-fixt
            ;;
        qwen-el-gpt4o)
            MODEL_PATH=/tmp/Qwen3-8B
            EXP_NAME=wildchat-el-q3-8b-rgpt4o
            ;;
        olmo-el-self)
            MODEL_PATH=/tmp/Olmo-3-7B-Instruct
            EXP_NAME=wildchat-el-om3-7b-r7b
            ;;
        olmo-el-gpt4o)
            MODEL_PATH=/tmp/Olmo-3-7B-Instruct
            EXP_NAME=wildchat-el-om3-7b-rgpt4o
            ;;
        qwen-rl-self)
            MODEL_PATH=/tmp/Qwen3-8B
            EXP_NAME=wildchat-rl-q3-8b-r8b
            ;;
        qwen-rl-gpt4o)
            MODEL_PATH=/tmp/Qwen3-8B
            EXP_NAME=wildchat-rl-q3-8b-rgpt4o
            ;;
        olmo-rl-self)
            MODEL_PATH=/tmp/Olmo-3-7B-Instruct
            EXP_NAME=wildchat-rl-om3-7b-r7b
            ;;
        olmo-rl-gpt4o)
            MODEL_PATH=/tmp/Olmo-3-7B-Instruct
            EXP_NAME=wildchat-rl-om3-7b-rgpt4o
            ;;
        *)
            echo "Unknown experiment: $1" >&2
            list_experiments >&2
            exit 2
            ;;
    esac
}

run_training() {
    case "$1" in
        qwen-el-self)
            bash scripts/train/train_el.sh \
                --model /tmp/Qwen3-8B \
                --exp_model_path /tmp/Qwen3-8B \
                --exp_name wildchat-el-q3-8b-r8b \
                --nnodes 4 \
                --rollout_n 8 \
                --kl_loss_type full \
                --kl_topk 256 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --rm_prompt_version v2 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True
            ;;
        qwen-el-self-iter)
            bash scripts/train/train_el.sh \
                --model /tmp/Qwen3-8B \
                --exp_model_path /tmp/Qwen3-8B \
                --exp_name wildchat-el-q3-8b-r8b-itert30-mopd025-fixt \
                --nnodes 4 \
                --rollout_n 8 \
                --kl_loss_type full \
                --kl_topk 256 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --rm_prompt_version v2 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True \
                --iter_teacher_steps 30 \
                --use_dataset wildchat,tulu3 \
                --multidata_ratio 1,0.25 \
                --multidata_rm_prompt v2,empty \
                --multidata_fixed_teacher 0,1
            ;;
        qwen-el-gpt4o)
            bash scripts/train/train_el.sh \
                --model /tmp/Qwen3-8B \
                --exp_model_path gpt4o \
                --use_rubric 4o \
                --exp_name wildchat-el-q3-8b-rgpt4o \
                --nnodes 4 \
                --rollout_n 8 \
                --kl_loss_type full \
                --kl_topk 256 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --rm_prompt_version v2 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True
            ;;
        olmo-el-self)
            bash scripts/train/train_el.sh \
                --model /tmp/Olmo-3-7B-Instruct \
                --exp_model_path /tmp/Olmo-3-7B-Instruct \
                --exp_name wildchat-el-om3-7b-r7b \
                --nnodes 4 \
                --rollout_n 8 \
                --kl_loss_type full \
                --kl_topk 256 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --rm_prompt_version v2 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True
            ;;
        olmo-el-gpt4o)
            bash scripts/train/train_el.sh \
                --model /tmp/Olmo-3-7B-Instruct \
                --exp_model_path gpt4o \
                --use_rubric 4o \
                --exp_name wildchat-el-om3-7b-rgpt4o \
                --nnodes 4 \
                --rollout_n 8 \
                --kl_loss_type full \
                --kl_topk 256 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --rm_prompt_version v2 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True
            ;;
        qwen-rl-self)
            bash scripts/train/train_rl.sh \
                --model /tmp/Qwen3-8B \
                --exp_model_path /tmp/Qwen3-8B \
                --exp_name wildchat-rl-q3-8b-r8b \
                --nnodes 4 \
                --rollout_n 8 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True
            ;;
        qwen-rl-gpt4o)
            bash scripts/train/train_rl.sh \
                --model /tmp/Qwen3-8B \
                --exp_model_path gpt4o \
                --use_rubric 4o \
                --exp_name wildchat-rl-q3-8b-rgpt4o \
                --nnodes 4 \
                --rollout_n 8 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True
            ;;
        olmo-rl-self)
            bash scripts/train/train_rl.sh \
                --model /tmp/Olmo-3-7B-Instruct \
                --exp_model_path /tmp/Olmo-3-7B-Instruct \
                --exp_name wildchat-rl-om3-7b-r7b \
                --nnodes 4 \
                --rollout_n 8 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True
            ;;
        olmo-rl-gpt4o)
            bash scripts/train/train_rl.sh \
                --model /tmp/Olmo-3-7B-Instruct \
                --exp_model_path gpt4o \
                --use_rubric 4o \
                --exp_name wildchat-rl-om3-7b-rgpt4o \
                --nnodes 4 \
                --rollout_n 8 \
                --actor_lr 1e-6 \
                --batch_size 256 \
                --max_prompt_length 8192 \
                --max_response_length 4096 \
                --gpu_memory_utilization 0.6 \
                --save_freq 10 \
                --save_optim True \
                --auto_resume True
            ;;
        *)
            echo "Unknown experiment: $1" >&2
            list_experiments >&2
            exit 2
            ;;
    esac
}

if [ "$#" -eq 0 ]; then
    usage
    exit 2
fi

ACTION=$1
case "$ACTION" in
    list)
        list_experiments
        ;;
    train)
        [ "$#" -eq 2 ] || { usage >&2; exit 2; }
        run_training "$2"
        ;;
    eval|eval_fuzzy|eval-fuzzy|eval_endtask|eval-ifeval)
        [ "$#" -eq 3 ] || { usage >&2; exit 2; }
        set_experiment_metadata "$2"
        CKPT=$3
        case "$ACTION" in
            eval)
                bash scripts/train/eval.sh \
                    --model "$MODEL_PATH" --exp_model_path gpt4o --exp_name "$EXP_NAME" \
                    --nnodes "${NNODES:-1}" --ckpt "$CKPT" --max_prompt_length 8192 \
                    --max_response_length 4096 --use_dataset wildchat_4o
                ;;
            eval_fuzzy|eval-fuzzy)
                bash scripts/train/eval_fuzzy.sh \
                    --model "$MODEL_PATH" --exp_name "$EXP_NAME" \
                    --nnodes 1 --ckpt "$CKPT"
                ;;
            eval_endtask|eval-ifeval)
                bash scripts/train/eval_endtask.sh \
                    --model "$MODEL_PATH" --exp_name "$EXP_NAME" \
                    --nnodes 1 --ckpt "$CKPT"
                ;;
        esac
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
