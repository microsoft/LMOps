BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 tools/convert_tokenization.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-name cc \
    --old-model-type mistral \
    --old-model-path $BASE_PATH/checkpoints/mistral/160M \
    --model-type fairseq \
    --model-path $BASE_PATH/checkpoints/fairseq/125M \
    --data-dir $BASE_PATH/processed_data/pretrain/cc/mistral-1025 \
    --save $BASE_PATH/processed_data/data_scorer_infer/ \
    --max-length 1024 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --chunk-num-per-shard 1000000