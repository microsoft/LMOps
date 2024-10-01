BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 tools/sample_dev_test.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-name cc-mistral-1025 \
    --data-dir processed_data/pretrain/cc/mistral-1025 \
    --save processed_data/pretrain_eval/ \
    --dev-num 100000 \
    --test-num 100000