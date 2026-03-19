# EXP_NAME,CKPT_START,CKPT_END,CKPT_STEP,RESUME_POLICY_NAME,RESUME_POLICY_CKPT,PROMPT_VERSION,VAL_SAMPLES_LIMIT,EXP_SEL_WITH_PREV,MAX_RESPONSE_LENGTH,TEXTGAME_NAME,TEXTGAME_MAX_RESPONSE,TEXTGAME_MAX_STEPS,TEXTGAME_NO_THINK,EXP_MODEL_PATH
if [ "$#" -gt 0 ]; then
    EXPERIMENTS=("$1")
else
    EXPERIMENTS=(
        "sokoban-q3-4b-ins-ext-v4-selwop,50,500,50,Qwen/Qwen3-4B-Instruct-2507,,v4,30,False,8192,Sokoban-v0,1024,5,True,"
        "frozenlake-q3-1b7-ext-v4-selwop,50,500,50,Qwen/Qwen3-1.7B,,v4,30,False,8192,FrozenLake-v0-raw,1024,5,False,"
    )
fi

for EXP_CONFIG in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r EXP_NAME CKPT_START CKPT_END CKPT_STEP RESUME_POLICY_NAME RESUME_POLICY_CKPT PROMPT_VERSION VAL_SAMPLES_LIMIT EXP_SEL_WITH_PREV MAX_RESPONSE_LENGTH TEXTGAME_NAME TEXTGAME_MAX_RESPONSE TEXTGAME_MAX_STEPS TEXTGAME_NO_THINK EXP_MODEL_PATH <<< "$EXP_CONFIG"


    echo "=========================================="
    echo "Processing experiment: $EXP_NAME"
    echo "Checkpoints: $CKPT_START to $CKPT_END, step $CKPT_STEP"
    echo "Resume Policy Name: $RESUME_POLICY_NAME"
    echo "Resume Policy Ckpt: $RESUME_POLICY_CKPT"
    echo "Prompt version: $PROMPT_VERSION"
    echo "Val samples limit: $VAL_SAMPLES_LIMIT"
    echo "Extract exp with previous exp in context: $EXP_SEL_WITH_PREV"
    echo "Max response length (used as max length for experiential knowledge): $MAX_RESPONSE_LENGTH"
    echo "Textgame name: $TEXTGAME_NAME"
    echo "Textgame max response length: $TEXTGAME_MAX_RESPONSE"
    echo "Textgame max steps: $TEXTGAME_MAX_STEPS"
    echo "Textgame no think: $TEXTGAME_NO_THINK"
    echo "Exp extractor model path: $EXP_MODEL_PATH"
    echo "=========================================="
    
    # Handle resume policy checkpoint
    if [[ "$RESUME_POLICY_NAME" == /* ]]; then
        MODEL_PATH=$RESUME_POLICY_NAME
        echo "Using huggingface model path: $MODEL_PATH"
    elif [ -n "${RESUME_POLICY_NAME}" ] && [ -n "${RESUME_POLICY_CKPT}" ]; then
        MODEL_PATH="/tmp/${RESUME_POLICY_NAME}/global_step_${RESUME_POLICY_CKPT}/actor/huggingface"
        mkdir -p $MODEL_PATH
        find /tmp/${RESUME_POLICY_NAME}/global_step_${RESUME_POLICY_CKPT}/actor/ -maxdepth 1 -type f ! -name "*.pt" -exec cp {} /tmp/${RESUME_POLICY_NAME}/global_step_${RESUME_POLICY_CKPT}/actor/huggingface/ \;
        python tools/merge_model2hf.py --local_dir /tmp/${RESUME_POLICY_NAME}/global_step_${RESUME_POLICY_CKPT}/actor
        echo "Using merged model path: $MODEL_PATH"
    else
        MODEL_PATH=$RESUME_POLICY_NAME
        echo "Warning: RESUME_POLICY_NAME is not a path and CKPT is missing. Using as is: $MODEL_PATH"
    fi
    
    CKPTS=()
    for ((i=CKPT_START; i<=CKPT_END; i+=CKPT_STEP)); do
        CKPTS+=($i)
    done
    echo "Checkpoints: ${CKPTS[@]}"
    
    for CKPT in "${CKPTS[@]}"; do
        bash scripts/textgame_extract.sh --model $MODEL_PATH --exp_name $EXP_NAME --nnodes 1 --ckpt $CKPT --prompt_version $PROMPT_VERSION --val_samples_limit $VAL_SAMPLES_LIMIT --exp_sel_with_prev $EXP_SEL_WITH_PREV --max_response_length "$MAX_RESPONSE_LENGTH" --textgame_name $TEXTGAME_NAME --textgame_max_response $TEXTGAME_MAX_RESPONSE --textgame_max_steps $TEXTGAME_MAX_STEPS --textgame_no_think $TEXTGAME_NO_THINK ${EXP_MODEL_PATH:+--exp_model_path "$EXP_MODEL_PATH"}
    done
done