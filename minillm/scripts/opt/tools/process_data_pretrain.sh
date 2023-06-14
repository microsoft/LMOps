BASE_PATH=${1}

MAX_LENGTH=512

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/roberta \
    --processed-data-dir ${BASE_PATH}/processed_data/roberta/opt/${MAX_LENGTH}/ \
    --model-path ${BASE_PATH}/checkpoints/opt-1.3B \
    --max-length ${MAX_LENGTH} \
    --train-num 20000000 \
    --data-process-workers 32 \
    --dev-num 10000 \