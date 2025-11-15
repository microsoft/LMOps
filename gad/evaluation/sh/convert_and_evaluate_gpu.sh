#!/bin/bash
set -x

# 参数检查
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    echo "Usage: $0 <eval_script_path> <base_checkpoint_path> <init_model_path> <template> [tp_size]"
    exit 1
fi

# 获取参数
eval_script_path=$1
base_checkpoint_path=$2
init_model_path=$3
template=$4
tp_size=${5:-1}  # 如果未提供第5个参数，默认为1
actor_dir="_actor"

# 获取可用的GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
NUM_GPU_GROUPS=$((NUM_GPUS / tp_size))  # 计算可用的GPU组数

# 函数：复制 tokenizer 文件
copy_tokenizer_files() {
    local ckpt_path=$1
    local init_model_path=$2
    local files_to_copy=(
        "added_tokens.json"
        "config.json"
        "generation_config.json"
        "special_tokens_map.json"
        "tokenizer_config.json"
        "tokenizer.json"
        "vocab.json"
    )
    # Add merges.txt to files_to_copy if it exists in source directory
    if [ -f "$init_model_path/merges.txt" ]; then
        files_to_copy+=("merges.txt")
    fi
    # 创建目标路径，确保它存在
    if [ ! -d "$ckpt_path" ]; then
        mkdir -p "$ckpt_path"
        echo "Created checkpoint directory: $ckpt_path" >&2
    else
        echo "Checkpoint directory already exists: $ckpt_path" >&2
    fi

    # 复制每个文件
    for filename in "${files_to_copy[@]}"; do
        src="$init_model_path/$filename"
        dst="$ckpt_path/$filename"
        if [ -e "$src" ]; then
            cp "$src" "$dst"
            echo "Copied $src to $dst"
        else
            echo "Warning: $src does not exist."
        fi
    done
}

# 函数：获取所有需要评估的检查点，并过滤掉已评估的
get_checkpoints_to_evaluate() {
    local base_path="$1"
    local checkpoints=()
    
    # 查找所有global_step*目录
    for ckpt_dir in "$base_path/$actor_dir"/global_step*; do
        if [ -d "$ckpt_dir" ]; then
            step_tag=$(basename "$ckpt_dir")
            checkpoints+=("$step_tag")
        fi
    done < <(find "$base_path/$actor_dir" -maxdepth 1 -type d -name "global_step*" -print0)
    
    # 正确输出数组内容
    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo ""
    else
        printf "%s\n" "${checkpoints[@]}"
    fi
}

# 函数：在指定GPU上处理单个检查点
process_checkpoint() {
    local step_tag=$1
    local group_id=$2
    
    # 计算该组的GPU ID范围
    local start_gpu=$((group_id * tp_size))
    local gpu_ids=""
    for ((i=0; i<tp_size; i++)); do
        if [ -n "$gpu_ids" ]; then
            gpu_ids="${gpu_ids},"
        fi
        gpu_ids="${gpu_ids}$((start_gpu + i))"
    done
    
    ckpt_path="$base_checkpoint_path/$actor_dir/$step_tag"
    output_file="$ckpt_path/pytorch_model.bin"
    done_file="$ckpt_path/convert_done.txt"
    original_dir=$(pwd)
    # echo "Converting checkpoint $step_tag on GPU $gpu_id" >&2

    if [ ! -f "$done_file" ]; then
        echo "Converting checkpoint $step_tag on GPU $gpu_ids" >&2
        cd "$base_checkpoint_path/$actor_dir/" || exit 1
        CUDA_VISIBLE_DEVICES=$gpu_ids python zero_to_fp32.py . "$output_file" -t "$step_tag"
        if [ $? -eq 0 ]; then
            copy_tokenizer_files "$ckpt_path" "$init_model_path"
            touch "$done_file"
        else
            echo "Error: Conversion failed for checkpoint $step_tag" >&2
            cd "$original_dir"
            return 1
        fi
    fi
    cd "$original_dir"
    echo "Evaluating checkpoint $step_tag on GPU $gpu_ids" >&2
    
    output_path="$base_checkpoint_path/$actor_dir/eval_results/$step_tag"
    mkdir -p "$output_path"
    
    CUDA_VISIBLE_DEVICES=$gpu_ids bash "$eval_script_path" ${template} "$ckpt_path" "$output_path"
}


readarray -t checkpoints_to_evaluate < <(get_checkpoints_to_evaluate "$base_checkpoint_path")

if [ ${#checkpoints_to_evaluate[@]} -eq 0 ]; then
    echo "No new checkpoints to evaluate." >&2
    exit 0
fi
# 检查GPU数量是否满足tp_size要求
if [ $((NUM_GPUS % tp_size)) -ne 0 ]; then
    echo "Error: Number of available GPUs ($NUM_GPUS) is not divisible by tp_size ($tp_size)" >&2
    exit 1
fi

echo "Found ${#checkpoints_to_evaluate[@]} checkpoints to evaluate:" >&2
printf '%s\n' "${checkpoints_to_evaluate[@]}" >&2

# 并行处理检查点，按GPU组分配
for i in "${!checkpoints_to_evaluate[@]}"; do
    group_id=$((i % NUM_GPU_GROUPS))
    step_tag="${checkpoints_to_evaluate[i]}"
    
    # 在后台启动处理任务
    process_checkpoint "$step_tag" "$group_id" &
    
    # 每启动NUM_GPU_GROUPS个任务后等待它们完成
    if [ $(((i + 1) % NUM_GPU_GROUPS)) -eq 0 ]; then
        wait
    fi
done


wait


echo "All conversions and evaluations completed."
