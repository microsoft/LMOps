BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 tools/sample_proxy_data.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-name cc-mistral-1025 \
    --data-dir processed_data/pretrain/cc/mistral-1025 \
    --save processed_data/proxy/ \
    --proxy-num 163840 \
    --max-state 10