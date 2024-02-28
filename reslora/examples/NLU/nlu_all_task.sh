export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0

exp='101'
method='reslora'
base_model="roberta-large"
debug_flag=0
seed=0
tasks=('cola' 'mrpc' 'rte' 'mnli' 'qnli' 'qqp' 'sst2' 'stsb' 'wnli')


for task in "${tasks[@]}"
do
  if [ $task == 'mnli' ]
  then
    num_train_epochs=5
  else
    num_train_epochs=20
  fi
  torchrun --nproc_per_node=$num_gpus examples/text-classification/run_glue.py \
  --model_name_or_path "/home/aiscuser/models/"$base_model \
  --task_name $task \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 3e-4 \
  --num_train_epochs $num_train_epochs \
  --output_dir '/home/aiscuser/train/exp_'$exp'/'$task'/model' \
  --overwrite_output_dir \
  --logging_steps 10 \
  --logging_dir '/home/aiscuser/train/exp_'$exp'/'$task'/log' \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --warmup_ratio 0.6 \
  --apply_lora \
  --lora_r 4 \
  --lora_alpha 16 \
  --seed $seed \
  --weight_decay 0.1 \
  --method $method \
  --res_flag 3 \
  --merge_flag 3 \
  --pre_num 4 \
  --debug_flag $debug_flag
done

