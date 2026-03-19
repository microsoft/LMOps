# EXP_NAME,CKPT_START,CKPT_END,CKPT_STEP,MODEL_PATH,PROMPT_VERSION,USE_BSL,EVAL_PREPEND_EXPERIENCE,EXP_PATH,MAX_RESPONSE_LENGTH,EXPERIENCE_MAX_LENGTH,SYSTEM_PROMPT_TYPE,SYSTEM_PROMPT_VERSION

exp_path_medmcqa=system_prompts/medmcqa.txt
exp_path_safety=system_prompts/safety.txt

EXPERIMENTS=(
    "sys-medmcqa-q25-7b-orgmodel,0,0,2,Qwen/Qwen2.5-7B-Instruct,v4,true,false,exp_path_medmcqa,512,4096,medmcqa,v1"
    "sys-medmcqa-q25-7b-orgwexp,0,0,2,Qwen/Qwen2.5-7B-Instruct,v4,true,true,exp_path_medmcqa,512,4096,medmcqa,v1"
    "sys-medmcqa-q25-7b,2,50,2,Qwen/Qwen2.5-7B-Instruct,v4,false,false,exp_path_medmcqa,512,4096,medmcqa,v1"
    "sys-medmcqa-q25-7b-offp,2,50,2,Qwen/Qwen2.5-7B-Instruct,v4,false,false,exp_path_medmcqa,512,4096,medmcqa,v1"
    
    "sys-safety-q25-7b-orgmodel,0,0,2,Qwen/Qwen2.5-7B-Instruct,v4,true,false,exp_path_safety,512,4096,safety,v1"
    "sys-safety-q25-7b-orgwexp,0,0,2,Qwen/Qwen2.5-7B-Instruct,v4,true,true,exp_path_safety,512,4096,safety,v1"
    "sys-safety-q25-7b,2,50,2,Qwen/Qwen2.5-7B-Instruct,v4,false,false,exp_path_safety,512,4096,safety,v1"
    "sys-safety-q25-7b-offp,2,50,2,Qwen/Qwen2.5-7B-Instruct,v4,false,false,exp_path_safety,512,4096,safety,v1"

)

for EXP_CONFIG in "${EXPERIMENTS[@]}"; do
    IFS=',' read -r EXP_NAME CKPT_START CKPT_END CKPT_STEP MODEL_PATH PROMPT_VERSION USE_BSL EVAL_PREPEND_EXPERIENCE EXP_PATH MAX_RESPONSE_LENGTH EXPERIENCE_MAX_LENGTH SYSTEM_PROMPT_TYPE SYSTEM_PROMPT_VERSION <<< "$EXP_CONFIG"
    
    USE_BSL=${USE_BSL:-false}
    EVAL_PREPEND_EXPERIENCE=${EVAL_PREPEND_EXPERIENCE:-false}

    if [[ $EXP_PATH == exp_path_* ]]; then
        EXP_PATH=${!EXP_PATH}
    fi

    echo "=========================================="
    echo "Processing experiment: $EXP_NAME"
    echo "Checkpoints: $CKPT_START to $CKPT_END, step $CKPT_STEP"
    echo "Model path: $MODEL_PATH"
    echo "Prompt version: $PROMPT_VERSION"
    echo "Use BSL: $USE_BSL"
    echo "Eval prepend experience: $EVAL_PREPEND_EXPERIENCE"
    echo "Exp path: $EXP_PATH"
    echo "Max response length: $MAX_RESPONSE_LENGTH"
    echo "Experience max length: $EXPERIENCE_MAX_LENGTH"
    echo "System prompt type: $SYSTEM_PROMPT_TYPE"
    echo "System prompt version: $SYSTEM_PROMPT_VERSION"
    echo "=========================================="
    
    CKPTS=()
    for ((i=CKPT_START; i<=CKPT_END; i+=CKPT_STEP)); do
        CKPTS+=($i)
    done
    echo "Generated checkpoints: ${CKPTS[@]}"
    
    for CKPT in "${CKPTS[@]}"; do
        bash scripts/sys_eval.sh --model $MODEL_PATH --exp_name $EXP_NAME --nnodes 1 --ckpt $CKPT --prompt_version $PROMPT_VERSION --use_bsl $USE_BSL --eval_prepend_experience $EVAL_PREPEND_EXPERIENCE --exp_path "$EXP_PATH" --max_response_length "$MAX_RESPONSE_LENGTH" --experience_max_length "$EXPERIENCE_MAX_LENGTH" --system_prompt_type "$SYSTEM_PROMPT_TYPE" --system_prompt_version "$SYSTEM_PROMPT_VERSION"
    done
done