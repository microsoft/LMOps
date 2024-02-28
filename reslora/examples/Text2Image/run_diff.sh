export num_gpus=8
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false
begin_exp=101
task_name='sd-pokemon'
methods=('lora' 'reslora1' 'reslora2')

export MODEL_NAME="stabilityai/stable-diffusion-2"
export DATASET_NAME="lambdalabs/pokemon-blip-captions"

for i in {0..2}
do
  exp=$((begin_exp+i))
  export output_dir='/home/train/exp_'$exp'/'$task_name
  accelerate launch --multi_gpu --gpu_ids "0,1,2,3,4,5,6,7" --main_process_port=29600 run_diffusion_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --method=${methods[$i]} \
    --image_column="image" \
    --caption_column="text" \
    --resolution=768 --random_flip \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=10 \
    --learning_rate=1e-4 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --seed=42 \
    --output_dir=$output_dir \
    --validation_prompt="a small bird with a black and white tail" --report_to="wandb" \
    --num_validation_images=4 \
    --validation_steps=20 \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=3 \
    --debug_flag=0
done



