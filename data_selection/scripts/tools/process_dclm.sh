BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data/dclm.py \
    --base-path ${BASE_PATH} \
    --type "tokenize" \
    --data-name "dclm" \
    --data-dir ${BASE_PATH}/downstream_data/dclm/ \
    --save ${BASE_PATH}/processed_data/ \
    --model-path ${BASE_PATH}/checkpoints/mistral/160M \
    --data-process-workers 32 \
    --max-length 1025 \
    --model-type mistral
