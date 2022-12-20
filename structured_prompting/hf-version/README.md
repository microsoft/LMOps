# Structured Prompting

## Environment
torch==1.13

transformers==git+https://github.com/huggingface/transformers

accelerate==0.15.0

bitsandbytes==0.35.4

datasets==2.7.1

## Example Usage
```
python3 eval_many.py \
  --model bloom \
  --dtype float16 \
  --parallel \ # activate model parallel
  --task sst2 \
  --strategy truncate \ # align method
  --data_path ./data \ # pretrained model saved in ./data/model/bloom
  --chunk_num 5 \ desired chunk number
  --max_length 2000
```
