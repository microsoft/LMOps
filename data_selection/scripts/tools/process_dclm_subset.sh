BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 tools/process_data/dclm_subset.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-name temp_dclm_subset0 \
    --model-path checkpoints/mistral/160M \
    --data-dir pretrain_data/dclm-baseline-1.0-subset0/ \
    --save processed_data/pretrain/ \
    --max-length 1025 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --model-type mistral \
    --chunk-num-per-shard 1000000 \
    --job-id 0 