export CUDA_VISIBLE_DEVICES="0"

# MODEL_NAME_OR_PATH="/mnt/msranlp/shaohanh/exp/simple-rl/checkpoints/Qwen2.5-Math-7B_ppo_from_base_math_lv35_4node/_actor/"
# MODEL_STEP="global_step4"
MODEL_NAME_OR_PATH=$1
MODEL_STEP=$2

# Qwen2.5-Math-Instruct Series
PROMPT_TYPE="qwen25-math-cot"
MODEL_ID="Qwen/Qwen2.5-Math-7B"

python $MODEL_NAME_OR_PATH/zero_to_fp32.py $MODEL_NAME_OR_PATH $MODEL_NAME_OR_PATH/$MODEL_STEP/pytorch_model.bin -t $MODEL_STEP

TARGET_DIR=$MODEL_NAME_OR_PATH/$MODEL_STEP/
mkdir -p "$TARGET_DIR"

FILES=(
  "merges.txt"
  "config.json"
  "generation_config.json"
  "tokenizer_config.json"
  "tokenizer.json"
  "vocab.json"
)
echo "start downloading model ${MODEL_ID} config files to ${TARGET_DIR} from Hugging Face..."
for file in "${FILES[@]}"; do
  URL="https://huggingface.co/${MODEL_ID}/resolve/main/${file}"
  echo "downloading ${file} from ${URL}"
  curl -L -o "${TARGET_DIR}/${file}" "${URL}"
done

bash sh/eval.sh $PROMPT_TYPE $TARGET_DIR $TARGET_DIR
