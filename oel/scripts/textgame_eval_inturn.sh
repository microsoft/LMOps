# EXP_NAME,CKPT_START,CKPT_END,CKPT_STEP,MODEL_PATH,USE_BSL,MAX_RESPONSE_LENGTH,TEXTGAME_NAME,TEXTGAME_MAX_STEPS,TEXTGAME_NO_THINK

if [ "$#" -gt 0 ]; then
    EXPERIMENTS=("$1")
else
    EXPERIMENTS=(
        
    )
fi

for EXP_CONFIG in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r EXP_NAME CKPT_START CKPT_END CKPT_STEP MODEL_PATH USE_BSL MAX_RESPONSE_LENGTH TEXTGAME_NAME TEXTGAME_MAX_STEPS TEXTGAME_NO_THINK <<< "$EXP_CONFIG"

    echo "=========================================="
    echo "Processing experiment: $EXP_NAME"
    echo "Checkpoints: $CKPT_START to $CKPT_END, step $CKPT_STEP"
    echo "Model path: $MODEL_PATH"
    echo "Use BSL: $USE_BSL"
    echo "Max response length: $MAX_RESPONSE_LENGTH"
    echo "Textgame name: $TEXTGAME_NAME"
    echo "Textgame max steps: $TEXTGAME_MAX_STEPS"
    echo "Textgame no think: $TEXTGAME_NO_THINK"
    echo "=========================================="
    
    CKPTS=()
    for ((i=CKPT_START; i<=CKPT_END; i+=CKPT_STEP)); do
        CKPTS+=($i)
    done
    echo "Generated checkpoints: ${CKPTS[@]}"
    
    for CKPT in "${CKPTS[@]}"; do
        bash scripts/textgame_eval.sh --model $MODEL_PATH --exp_name $EXP_NAME --nnodes 1 --ckpt $CKPT --use_bsl $USE_BSL --max_response_length "$MAX_RESPONSE_LENGTH" --textgame_name $TEXTGAME_NAME --textgame_max_steps $TEXTGAME_MAX_STEPS --textgame_no_think $TEXTGAME_NO_THINK
    done
done