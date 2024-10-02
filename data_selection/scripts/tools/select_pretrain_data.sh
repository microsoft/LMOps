BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}

python3 tools/select_pretrain_data.py \
    --base-path $BASE_PATH \
    --type data_processing \
    --data-dir  $BASE_PATH/processed_data/pretrain/cc/mistral-1025 \
    --save $BASE_PATH/processed_data/pretrain/ \
    --model-type mistral \
    --model-path $BASE_PATH/checkpoints/mistral/160M \
    --data-scorer-model-type fairseq \
    --data-scorer-tokenizer-path $BASE_PATH/checkpoints/fairseq/125M \
    --ds-score-path $BASE_PATH/results/data_scorer_infer/cc/cc-sgd100-160M-10k-lima \
    --ds-ratio 0.4 \
    --ds-gumbel-temperature 0.1 \
    --data-name cc-sgd100-160M-10k-lima \