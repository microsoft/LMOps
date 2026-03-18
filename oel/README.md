# Online Experiential Learning for Language Models

This repository contains the implementation for our paper **"Online Experiential Learning for Language Models"**.

📄 **Paper**: [arXiv:2603.16856](https://arxiv.org/abs/2603.16856)

The code is built on [VeRL](https://github.com/volcengine/verl). We provide online experiential learning code for two environments: Frozen Lake and Sokoban.

[On-Policy Context Distillation](https://arxiv.org/abs/2602.12275) is our preceding work. We open-source its code at [OPCD-Code](https://github.com/microsoft/LMOps/tree/main/opcd), which includes mathematical reasoning, text-based game tasks for experiential knowledge distillation, and system prompt distillation. Off-policy context distillation is also implemented in that codebase. Feel free to refer to it if needed.

## 🚀 Environment Setup


If you use A100, H100 or H200:

```bash
bash run_docker.sh
cd /tmp ; git clone --depth 1 https://github.com/microsoft/LMOps.git
cd /tmp/LMOps/oel
bash setup.sh
bash ray_node_setup.sh
```

If you use B200:

```bash
bash run_docker_b200.sh
cd /tmp ; git clone --depth 1 https://github.com/microsoft/LMOps.git
cd /tmp/LMOps/oel
bash setup_b200.sh
bash ray_node_setup.sh
source .venv/bin/activate
```

## 📦 Usage

First login your wandb account:

```bash
export WANDB_PROJECT=${YOUR_WANDB_PROJECT} ; export WANDB_API_KEY=${YOUR_WANDB_KEY}
```

### Sokoban, Qwen3-4B-Instruct-2507 (non-thinking model)

#### Round 1

```bash
# 1. Experiential knowledge extraction.
# The 'CKPT' passed in actually represents different random seed.
# You can split them to multiples jobs.
# We accumulate 100 steps (to show the saturation) and use 50 steps to consolidate.
# No training, just inference.
bash scripts/textgame_extract_inturn.sh "oel-sokoban-q3-4b-ins-ext-v4-selwp-round1,50,500,50,Qwen/Qwen3-4B-Instruct-2507,,v4,100,True,8192,Sokoban-v0,1024,5,True,1," ; python tools/make_exp_list.py "oel-sokoban-q3-4b-ins-ext-v4-selwp-round1,50,500,50,100,50"
```

```bash
# 2. Collection of user trajectories, to construct partial rollouts later.
# No training, just inference.
bash scripts/textgame_generate_deploy.sh --model Qwen/Qwen3-4B-Instruct-2507 --exp_name oel-sokoban-q3-4b-ins-round1-deploy --nnodes 1 --oel_round 1 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --total_training_steps 100
```

```bash
# 3. Experiential knowledge consolidation.
# Training.
bash scripts/textgame_consolidate.sh --model Qwen/Qwen3-4B-Instruct-2507 --exp_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1 --nnodes 2 --oel_round 1 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --deploy_save_dir /tmp/oel-sokoban-q3-4b-ins-round1-deploy/deploy_data --exp_path /tmp/oel-sokoban-q3-4b-ins-ext-v4-selwp-round1/experience_list.txt --total_training_steps 100 --save_freq 2
```

```bash
# (Optional) Evaluate checkpoints during consolidation.
# No training, just inference.
bash scripts/textgame_eval_inturn.sh "oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1,2,100,2,Qwen/Qwen3-4B-Instruct-2507,false,1024,Sokoban-v0,5,true"
```



#### Round 2

```bash
# 1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-sokoban-q3-4b-ins-ext-v4-selwp-round2,50,500,50,oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1,100,v4,100,True,8192,Sokoban-v0,1024,5,True,2," ; python tools/make_exp_list.py "oel-sokoban-q3-4b-ins-ext-v4-selwp-round2,50,500,50,100,50"
```

```bash
# 2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --resume_policy_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1 --resume_policy_ckpt 100 --exp_name oel-sokoban-q3-4b-ins-round2-deploy --nnodes 1 --oel_round 2 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --total_training_steps 100
```

```bash
# 3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --resume_policy_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round1 --resume_policy_ckpt 100 --exp_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round2 --nnodes 2 --oel_round 2 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --deploy_save_dir /tmp/oel-sokoban-q3-4b-ins-round2-deploy/deploy_data --exp_path /tmp/oel-sokoban-q3-4b-ins-ext-v4-selwp-round2/experience_list.txt --total_training_steps 100 --save_freq 2
```

#### Round 3

```bash
# 1. Experiential knowledge extraction.
bash scripts/textgame_extract_inturn.sh "oel-sokoban-q3-4b-ins-ext-v4-selwp-round3,50,500,50,oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round2,100,v4,100,True,8192,Sokoban-v0,1024,5,True,3," ; python tools/make_exp_list.py "oel-sokoban-q3-4b-ins-ext-v4-selwp-round3,50,500,50,100,50"
```

```bash
# 2. Collection of user trajectories, to construct partial rollouts later.
bash scripts/textgame_generate_deploy.sh --resume_policy_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round2 --resume_policy_ckpt 100 --exp_name oel-sokoban-q3-4b-ins-round3-deploy --nnodes 1 --oel_round 3 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --total_training_steps 100
```

```bash
# 3. Experiential knowledge consolidation.
bash scripts/textgame_consolidate.sh --resume_policy_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round2 --resume_policy_ckpt 100 --exp_name oel-sokoban-q3-4b-ins-v4-selwp-lr1e-6-round3 --nnodes 2 --oel_round 3 --kl_loss_type full --kl_topk 256 --actor_lr 1e-6 --experience_max_length 8192 --textgame_name Sokoban-v0 --max_response_length 1024 --textgame_max_steps 5 --textgame_no_think True --deploy_save_dir /tmp/oel-sokoban-q3-4b-ins-round3-deploy/deploy_data --exp_path /tmp/oel-sokoban-q3-4b-ins-ext-v4-selwp-round3/experience_list.txt --total_training_steps 100 --save_freq 2
```



### Frozen Lake, Qwen3-1.7B, Qwen3-4B, Qwen3-8B  (thinking model)
See `usage_example.sh`.




## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{ye2026onlineexperientiallearninglanguage,
    title={Online Experiential Learning for Language Models}, 
    author={Tianzhu Ye and Li Dong and Qingxiu Dong and Xun Wu and Shaohan Huang and Furu Wei},
    year={2026},
    eprint={2603.16856},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2603.16856}, 
}
```