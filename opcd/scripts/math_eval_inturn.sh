# EXP_NAME,CKPT_START,CKPT_END,CKPT_STEP,MODEL_PATH,USE_BSL,MAX_RESPONSE_LENGTH

EXPERIMENTS=(
    "math-q3-8b-lr5e-6,2,50,2,Qwen/Qwen3-8B,false,16384"
)

for EXP_CONFIG in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r EXP_NAME CKPT_START CKPT_END CKPT_STEP MODEL_PATH USE_BSL MAX_RESPONSE_LENGTH <<< "$EXP_CONFIG"
    
    echo "=========================================="
    echo "Processing experiment: $EXP_NAME"
    echo "Checkpoints: $CKPT_START to $CKPT_END, step $CKPT_STEP"
    echo "Model path: $MODEL_PATH"
    echo "Use BSL: $USE_BSL"
    echo "Max response length: $MAX_RESPONSE_LENGTH"
    echo "=========================================="
    
    CKPTS=()
    for ((i=CKPT_START; i<=CKPT_END; i+=CKPT_STEP)); do
        CKPTS+=($i)
    done
    echo "Generated checkpoints: ${CKPTS[@]}"
    
    for CKPT in "${CKPTS[@]}"; do
        bash scripts/math_eval.sh --model $MODEL_PATH --exp_name $EXP_NAME --nnodes 1 --ckpt $CKPT --use_bsl $USE_BSL --max_response_length "$MAX_RESPONSE_LENGTH"
    done
done