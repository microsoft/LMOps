#!/bin/bash

# 设置模型路径
MODEL_NAME_OR_PATH="/mnt/msranlp/shaohanh/exp/simple-rl/checkpoints/Qwen2.5-Math-7B_ppo_from_base_math_lv35_4node/_actor/"
STAET_STEP=4
END_STEP=60
STEP_INTERVAL=4

# 循环遍历步数
for i in $(seq $STAET_STEP $STEP_INTERVAL $END_STEP)
do
    echo "Evaluating model at step $i"
    MODEL_STEP="global_step$i"

    # 设置CUDA设备
    export CUDA_VISIBLE_DEVICES="0"

    # Qwen2.5-Math-Instruct Series
    PROMPT_TYPE="qwen25-math-cot"
    MODEL_ID="Qwen/Qwen2.5-Math-7B"

    # 合并权重文件
    python $MODEL_NAME_OR_PATH/zero_to_fp32.py $MODEL_NAME_OR_PATH $MODEL_NAME_OR_PATH/$MODEL_STEP/pytorch_model.bin -t $MODEL_STEP

    # 创建目标目录
    TARGET_DIR=$MODEL_NAME_OR_PATH/$MODEL_STEP/
    mkdir -p "$TARGET_DIR"

    # 下载配置文件
    FILES=(
      "merges.txt"
      "config.json"
      "generation_config.json"
      "tokenizer_config.json"
      "tokenizer.json"
      "vocab.json"
    )
    echo "Start downloading model ${MODEL_ID} config files to ${TARGET_DIR} from Hugging Face..."
    for file in "${FILES[@]}"; do
      URL="https://huggingface.co/${MODEL_ID}/resolve/main/${file}"
      echo "Downloading ${file} from ${URL}"
      curl -L -o "${TARGET_DIR}/${file}" "${URL}"
    done
done
