export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
begin_exp=101
task_name='svamp'
merge_flags=(0 3 4)


for i in {0..2}
do
  echo '==============================begin new exp====================================='
  echo $i
  exp="$((begin_exp+i))"
  export output_dir='/home/train/exp_'$exp'/'$task_name
  torchrun --nproc_per_node=$num_gpus \
  examples/reason/run_nlg.py \
  --fp16 \
  --deepspeed ../ds_config_stage1.json \
  --model_name_or_path "/home/models/Llama-2-7b-hf" \
  --task_name $task_name \
  --res_flag 1 \
  --merge_flag ${merge_flags[$i]} \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --max_source_length 500 \
  --max_target_length 500 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 0 \
  --output_dir $output_dir/model \
  --result_path $output_dir/result \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir $output_dir/log \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 1 \
  --warmup_ratio 0.06 \
  --lora_r 4 \
  --lora_alpha 8 \
  --seed 0 \
  --weight_decay 0.1 \
  --method "reslora" \
  --debug_flag 0
done

