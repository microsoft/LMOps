MODEL_NAME_OR_PATH="/mnt/msranlp/shaohanh/exp/simple-rl/checkpoints/Qwen2.5-Math-7B_ppo_from_base_math_lv35_4node/_actor/"

for i in {8..60..4}
do
    echo "evaluating model at step $i"
    # MODEL_STEP="global_step4"
    MODEL_STEP="global_step$i"
    bash run_eval.sh $MODEL_NAME_OR_PATH $MODEL_STEP
done
