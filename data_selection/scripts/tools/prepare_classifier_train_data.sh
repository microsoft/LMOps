BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 tools/prepare_data_scorer_train_data.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-dir $BASE_PATH/processed_data/proxy/cc-mistral-1025/163840 \
    --model-type mistral \
    --model-path $BASE_PATH/checkpoints/mistral/160M \
    --proxy-score-path $BASE_PATH/results/pmp_solver/cc-lima/160M-10k/sgd-t100-bs8-lr0.008constant1e-07-G2-N16-NN2ct10 \
    --data-scorer-tokenizer-path $BASE_PATH/checkpoints/fairseq/125M \
    --data-scorer-model-type fairseq \
    --data-name cc-sgd100-160M-10k-lima \
    --save $BASE_PATH/processed_data/data_scorer_train/ \
    --max-length 1024 \
    --log-interval 10000 \
    --data-process-workers 32 \
    --chunk-num-per-shard 1000000 \
    --dev-num 16384 \
    --proxy-num 163840