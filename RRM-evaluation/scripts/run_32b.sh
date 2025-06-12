export CUDA_VISIBLE_DEVICES=0,1


seed=$1

round_num=$2 

run_name="MATH_32b_round_${round_num}_seed_${seed}_with_shuffle"
#run_name="MMLU_32b_round_${round_num}_seed_${seed}_with_shuffle"
#run_name="GPQA_32b_round_${round_num}_seed_${seed}_with_shuffle"
#run_name="RewardBench_32b_round_${round_num}_seed_${seed}_with_shuffle"


echo "Running with run_name=${run_name}"

python run_generative.py \
    --model "Reward-Reasoning/RRM-32B" \
    --model_modifier "Skywork" \
    --dataset "MATH" \
    --max_response_tokens 8192 \
    --temperature 0.6 \
    --run_name "$run_name" \
    --seed "$seed" \
    --num_gpus 2 \
