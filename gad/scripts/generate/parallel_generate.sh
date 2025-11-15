val_datas=("lmsys" "dolly" "self-inst" "Vicuna")
for val_data in "${val_datas[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/generate/generate.sh --model /tmp/Qwen2.5-7B-Instruct --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 --val_data ${val_data} --ckpt_start 800 --ckpt_end 1200 --ckpt_step 50 --nnodes 1 --ngpus 2 &
    CUDA_VISIBLE_DEVICES=2,3 bash scripts/generate/generate.sh --model /tmp/Qwen2.5-7B-Instruct --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 --val_data ${val_data} --ckpt_start 1250 --ckpt_end 1600 --ckpt_step 50 --nnodes 1 --ngpus 2 &
    CUDA_VISIBLE_DEVICES=4,5 bash scripts/generate/generate.sh --model /tmp/Qwen2.5-7B-Instruct --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 --val_data ${val_data} --ckpt_start 1650 --ckpt_end 2000 --ckpt_step 50 --nnodes 1 --ngpus 2 &
    CUDA_VISIBLE_DEVICES=6,7 bash scripts/generate/generate.sh --model /tmp/Qwen2.5-7B-Instruct --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 --val_data ${val_data} --ckpt_start 2050 --ckpt_end 2400 --ckpt_step 50 --nnodes 1 --ngpus 2 &
    wait
done