BASE_PATH=${1}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_dolly.py \
    --data-dir ${BASE_PATH}/results/gpt2/gen/gpt2-xlarge-sft/t1.0-l512 \
    --processed-data-dir ${BASE_PATH}/processed_data/dolly/pseudo \
    --model-path ${BASE_PATH}/checkpoints/gpt2-large \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num -1 \
    --model-type gpt2-xlarge-sft

cp ${BASE_PATH}/processed_data/dolly/full/gpt2/valid_0.bin ${BASE_PATH}/processed_data/dolly/pseudo/gpt2-xlarge-sft/
cp ${BASE_PATH}/processed_data/dolly/full/gpt2/valid_0.idx ${BASE_PATH}/processed_data/dolly/pseudo/gpt2-xlarge-sft/
cp ${BASE_PATH}/processed_data/dolly/full/gpt2/valid.jsonl ${BASE_PATH}/processed_data/dolly/pseudo/gpt2-xlarge-sft/
