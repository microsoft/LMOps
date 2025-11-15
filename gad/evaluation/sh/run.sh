bash sh/eval_single_node.sh \
    --run_name Qwen2.5-Math-7B_ppo_from_base_math_lv35  \
    --init_model_path model_hub/models--Qwen--Qwen2.5-Math-7B/snapshots/b101308fe89651ea5ce025f25317fea6fc07e96e \
    --template qwen25-math-cot  \
    --tp_size 2