# On-Policy Context Distillation for Language Models

This repository contains the implementation for our paper **"On-Policy Context Distillation for Language Models"**.

📄 **Paper**: [arXiv:2602.12275](https://arxiv.org/abs/2602.12275)

The code is built on [VeRL](https://github.com/volcengine/verl). We provide on-policy context distillation code for mathematical reasoning (on DAPO dataset), text-based games (Frozen Lake and Sokoban), and system prompt distillation (medical and safety system prompts).

[Online Experiential Learning](https://arxiv.org/abs/2603.16856) is our follow-up work. We open-source its code at [OEL-Code](https://github.com/microsoft/LMOps/tree/main/oel). Feel free to refer to it if needed.


## 🚀 Environment Setup

If you use A100, H100 or H200:

```bash
bash run_docker.sh
cd /tmp ; git clone --depth 1 https://github.com/microsoft/LMOps.git
cd /tmp/LMOps/opcd
bash setup.sh
bash ray_node_setup.sh
```

If you use B200:

```bash
bash run_docker_b200.sh
cd /tmp ; git clone --depth 1 https://github.com/microsoft/LMOps.git
cd /tmp/LMOps/opcd
bash setup_b200.sh
bash ray_node_setup.sh
source .venv/bin/activate
```

## 🧪 Data Preparation

```bash
python tools/prepare_data.py
```

## 📖 Code Walkthrough

Main Entrance: [Ray Trainer](https://github.com/microsoft/LMOps/blob/main/opcd/verl/verl/trainer/ppo/ray_trainer.py#L2100)

Rollout: [Rollout](https://github.com/microsoft/LMOps/blob/main/opcd/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L520) and [TextGame Rollout](https://github.com/microsoft/LMOps/blob/main/opcd/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py#L682)

Update Policy: [Update](https://github.com/microsoft/LMOps/blob/main/opcd/verl/verl/workers/actor/dp_actor.py#L533) and [Reverse KL](https://github.com/microsoft/LMOps/blob/main/opcd/verl/verl/trainer/ppo/core_algos.py#L831)


## 📦 Usage

First login your wandb account:

```bash
export WANDB_PROJECT=${YOUR_WANDB_PROJECT} ; export WANDB_API_KEY=${YOUR_WANDB_KEY}
```

### Mathematical Reasoning (Experiential Knowledge Distillation)

#### On-Policy Context Distillation

Experiential knowledge extraction
```bash
bash scripts/math_extract_inturn.sh
```

Find the experiential knowledge that leads to the highest accuracy on validation during extraction.
```bash
export BEST_EXP_PATH={YOUR_BEST_EXP_PATH}
```

Experiential knowledge consolidation
```bash
bash scripts/math_consolidate.sh --model Qwen/Qwen3-8B --exp_name math-q3-8b-lr5e-6 --nnodes 2 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 16384 --exp_path ${BEST_EXP_PATH}
```

Evaluate checkpoints during consolidation
```bash
bash scripts/math_eval_inturn.sh
```

You can also use different size between teacher and student. Qwen3-8B teacher distill Qwen3-4B student
```bash
bash scripts/math_consolidate.sh --model Qwen/Qwen3-4B --ref_model_path /tmp/Qwen3-8B --exp_name math-q3-8b-dist-4b-lr5e-6 --nnodes 2 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 16384 --exp_path ${BEST_EXP_PATH}
```

#### Baseline: Context Distillation (off-policy)
After extraction, first use teacher with knowledge in context to inference and save trajectories and logits
```bash
bash scripts/math_generate_offp.sh --model Qwen/Qwen3-8B --exp_name math-q3-8b-offp-data --nnodes 2 --rollout_n 1 --kl_loss_type full --kl_topk 256 --max_response_length 16384 --exp_path ${BEST_EXP_PATH}
```

Then use it to context distill the student (off-policy)
```bash
bash scripts/math_train_offp.sh --model Qwen/Qwen3-8B --exp_name math-q3-8b-lr5e-6-offp --nnodes 2 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --exp_path ${BEST_EXP_PATH} --off_policy_save_dir /tmp/math-q3-8b-offp-data/off_policy_data --max_response_length 16384
```




### Text Games (Experiential Knowledge Distillation)

#### On-Policy Context Distillation

Experiential knowledge extraction.
```bash
bash scripts/textgame_extract_inturn.sh
```

Find the experiential knowledge that leads to the highest pass rate on validation during extraction.
```bash
export BEST_EXP_PATH_SOKOBAN={YOUR_BEST_EXP_PATH_SOKOBAN}
export BEST_EXP_PATH_FROZENLAKE={YOUR_BEST_EXP_PATH_FROZENLAKE}
```

Experiential knowledge consolidation
```bash
bash scripts/textgame_consolidate.sh --model Qwen/Qwen3-4B-Instruct-2507 --exp_name sokoban-q3-4b-ins-lr5e-6 --nnodes 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 1024 --experience_max_length 8192 --textgame_name Sokoban-v0 --textgame_max_steps 5 --textgame_no_think True --exp_path ${BEST_EXP_PATH_SOKOBAN}

bash scripts/textgame_consolidate.sh --model Qwen/Qwen3-1.7B --exp_name frozenlake-q3-1b7-think-lr5e-6 --nnodes 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 1024 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --textgame_max_steps 5 --textgame_no_think False --exp_path ${BEST_EXP_PATH_FROZENLAKE}
```

Evaluate checkpoints during consolidation
```bash
bash scripts/textgame_eval_inturn.sh
```

#### Baseline: Context Distillation (off-policy)
After extraction, first use teacher with knowledge in context to inference and save trajectories and logits
```bash
bash scripts/textgame_generate_offp.sh --model Qwen/Qwen3-4B-Instruct-2507 --exp_name sokoban-q3-4b-ins-offp-data --nnodes 1 --kl_loss_type full --kl_topk 256 --max_response_length 1024 --experience_max_length 8192 --textgame_name Sokoban-v0 --textgame_max_steps 5 --textgame_no_think True --exp_path ${BEST_EXP_PATH_SOKOBAN}

bash scripts/textgame_generate_offp.sh --model Qwen/Qwen3-1.7B --exp_name frozenlake-q3-1b7-think-offp-data --nnodes 1 --kl_loss_type full --kl_topk 256 --max_response_length 1024 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --textgame_max_steps 5 --textgame_no_think False --exp_path ${BEST_EXP_PATH_FROZENLAKE}
```

Then use it to context distill the student (off-policy)
```bash
bash scripts/textgame_train_offp.sh --model Qwen/Qwen3-4B-Instruct-2507 --exp_name sokoban-q3-4b-ins-lr5e-6-offp --nnodes 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 1024 --experience_max_length 8192 --textgame_name Sokoban-v0 --textgame_max_steps 5 --textgame_no_think True --exp_path ${BEST_EXP_PATH_SOKOBAN} --off_policy_save_dir /tmp/sokoban-q3-4b-ins-offp-data/off_policy_data

bash scripts/textgame_train_offp.sh --model Qwen/Qwen3-1.7B --exp_name frozenlake-q3-1b7-think-lr5e-6-offp --nnodes 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 1024 --experience_max_length 8192 --textgame_name FrozenLake-v0-raw --textgame_max_steps 5 --textgame_no_think False --exp_path ${BEST_EXP_PATH_FROZENLAKE} --off_policy_save_dir /tmp/frozenlake-q3-1b7-think-offp-data/off_policy_data
```





### System Prompt Distillation

#### On-Policy Context Distillation

Experiential knowledge consolidation
```bash
# medical
bash scripts/sys_consolidate.sh --model meta-llama/Llama-3.1-8B-Instruct --exp_name sys-medmcqa-l31-8b --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type medmcqa --exp_path system_prompts/medmcqa.txt --total_training_steps 50 --save_freq 2

bash scripts/sys_consolidate.sh --model meta-llama/Llama-3.2-3B-Instruct --exp_name sys-medmcqa-l32-3b --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type medmcqa --exp_path system_prompts/medmcqa.txt --total_training_steps 50 --save_freq 2

bash scripts/sys_consolidate.sh --model Qwen/Qwen2.5-7B-Instruct --exp_name sys-medmcqa-q25-7b --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type medmcqa --exp_path system_prompts/medmcqa.txt --total_training_steps 50 --save_freq 2


# safety
bash scripts/sys_consolidate.sh --model meta-llama/Llama-3.1-8B-Instruct --exp_name sys-safety-l31-8b --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --kl_renorm_topk True --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type safety --exp_path system_prompts/safety.txt --total_training_steps 50 --save_freq 2

bash scripts/sys_consolidate.sh --model meta-llama/Llama-3.2-3B-Instruct --exp_name sys-safety-l32-3b --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type safety --exp_path system_prompts/safety.txt --total_training_steps 50 --save_freq 2

bash scripts/sys_consolidate.sh --model Qwen/Qwen2.5-7B-Instruct --exp_name sys-safety-q25-7b --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type safety --exp_path system_prompts/safety.txt --total_training_steps 50 --save_freq 2
```

You also can use different sizes between teacher and student
```bash
bash scripts/sys_consolidate.sh --model Qwen/Qwen2.5-3B-Instruct --ref_model_path Qwen/Qwen2.5-7B-Instruct --exp_name sys-safety-q25-7b-dist-3b --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type safety --exp_path system_prompts/safety.txt --total_training_steps 50 --save_freq 2
```

Evaluate checkpoints during consolidation
```bash
bash scripts/sys_eval_inturn.sh
```

#### Baseline: Context Distillation (off-policy)

```bash
bash scripts/sys_generate_offp.sh.sh --model Qwen/Qwen2.5-7B-Instruct --exp_name sys-medmcqa-q25-7b-offp-data --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type medmcqa --exp_path system_prompts/medmcqa.txt   ;   bash scripts/sys_train_offp.sh --model Qwen/Qwen2.5-7B-Instruct --exp_name sys-medmcqa-q25-7b-offp --nnodes 1 --rollout_n 1 --kl_loss_type full --kl_topk 256 --actor_lr 5e-6 --max_response_length 512 --experience_max_length 4096 --system_prompt_type medmcqa --exp_path system_prompts/medmcqa.txt --off_policy_save_dir /tmp/sys-medmcqa-q25-7b-offp-data/off_policy_data --total_training_steps 50 --save_freq 2
```



## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{ye2026onpolicycontextdistillationlanguage,
      title={On-Policy Context Distillation for Language Models}, 
      author={Tianzhu Ye and Li Dong and Xun Wu and Shaohan Huang and Furu Wei},
      year={2026},
      eprint={2602.12275},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.12275}, 
}
```