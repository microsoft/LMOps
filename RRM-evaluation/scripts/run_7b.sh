export CUDA_VISIBLE_DEVICES=0,1


seed=$1

round_num=$2 

dataset=$3

run_name="MATH_7b_round_${round_num}_seed_${seed}_with_shuffle"
#run_name="MMLU_7b_round_${round_num}_seed_${seed}_with_shuffle"
#run_name="GPQA_7b_round_${round_num}_seed_${seed}_with_shuffle"
#run_name="RewardBench_7b_round_${round_num}_seed_${seed}_with_shuffle"

echo "Running with run_name=${run_name}"

python run_generative.py \
    --model "Reward-Reasoning/RRM-7B" \
    --model_modifier "Skywork" \
    --dataset "$dataset" \
    --max_response_tokens 8192 \
    --temperature 0.6 \
    --run_name "$run_name" \
    --seed "$seed" \
    --num_gpus 2 \
