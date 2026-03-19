# EXP_NAME,CKPT_START,CKPT_END,CKPT_STEP,MODEL_PATH,PROMPT_VERSION,VAL_SAMPLES_LIMIT,EXP_SEL_WITH_PREV,MAX_RESPONSE_LENGTH
EXPERIMENTS=(
    "math-q3-8b-ext-v4-selwop,50,500,50,Qwen/Qwen3-8B,v4,30,False,16384"
)

for EXP_CONFIG in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r EXP_NAME CKPT_START CKPT_END CKPT_STEP MODEL_PATH PROMPT_VERSION VAL_SAMPLES_LIMIT EXP_SEL_WITH_PREV MAX_RESPONSE_LENGTH <<< "$EXP_CONFIG"

    echo "=========================================="
    echo "Processing experiment: $EXP_NAME"
    echo "Checkpoints: $CKPT_START to $CKPT_END, step $CKPT_STEP"
    echo "Model path: $MODEL_PATH"
    echo "Prompt version: $PROMPT_VERSION"
    echo "Val samples limit: $VAL_SAMPLES_LIMIT"
    echo "Exp sel with prev: $EXP_SEL_WITH_PREV"
    echo "Max response length: $MAX_RESPONSE_LENGTH"
    echo "=========================================="
    
    CKPTS=()
    for ((i=CKPT_START; i<=CKPT_END; i+=CKPT_STEP)); do
        CKPTS+=($i)
    done
    echo "Generated checkpoints: ${CKPTS[@]}"
    
    for CKPT in "${CKPTS[@]}"; do
        bash scripts/math_extract.sh --model $MODEL_PATH --exp_name $EXP_NAME --nnodes 1 --ckpt $CKPT --prompt_version $PROMPT_VERSION --val_samples_limit $VAL_SAMPLES_LIMIT --exp_sel_with_prev $EXP_SEL_WITH_PREV --max_response_length "$MAX_RESPONSE_LENGTH"
    done
done