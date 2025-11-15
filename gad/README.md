# Black-Box On-Policy Distillation of Large Language Models

This repository contains the implementation and resources for our paper **"Black-Box On-Policy Distillation of Large Language Models"**.

📄 **Paper**: [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)

## 📋 Todo List

- [x] **Code**: Provided
- [ ] **Data**: Will be provided before November 30, 2025
- [ ] **Model Checkpoint**: Will be provided before November 30, 2025

## 🚀 Getting Started

### Docker Environment

We use `czwin32768/verl2:v0.2.0-vllm085` which has `vllm==0.8.5` as the recommended docker image.

### Environment Setup

```bash
bash local_docker.sh
cd /tmp
git clone https://github.com/microsoft/LMOps.git
cd /tmp/LMOps/gad
git clone https://github.com/YTianZHU/verl.git
bash local_setup.sh
```

### Implementation

We use [VeRL](https://github.com/volcengine/verl) as the implementation codebase.
**We hack the use of critic module as our discriminator**.

## 📦 Training

### Baseline: SeqKD

To run the baseline SeqKD:

```bash
cd verl
git checkout seqkd
cd ..
bash scripts/train/gpt5-chat-filtered-7b-seqkd-lr5e-6.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-seqkd-lr5e-6 \
  --nnodes 1
```

### Generative Adversarial Distillation

#### Stage 1: Warmup (1 epoch, ~800 steps)

To run the warmup stage of training:

```bash
cd verl
git checkout warmup
cd ..
bash scripts/train/gpt5-chat-filtered-7b-warmup-lr1e-6.sh \
  --model /tmp/Qwen2.5-7B-Instruct \
  --reward_model /tmp/Qwen2.5-7B-Instruct \
  --exp_name gpt5-chat-filtered-7b-warmup-lr1e-6 \
  --nnodes 1
```

#### Stage 2: GAD Training Stage

To run GAD training stage:

```bash
cd verl
git checkout gad
cd ..
STEP=800
mkdir /tmp/gpt5-chat-filtered-7b-adversarial-lr1e-6
cp -r /tmp/gpt5-chat-filtered-7b-warmup-lr1e-6/global_step_${STEP} \
  /tmp/gpt5-chat-filtered-7b-adversarial-lr1e-6/
echo ${STEP} > /tmp/gpt5-chat-filtered-7b-adversarial-lr1e-6/latest_checkpointed_iteration.txt
bash scripts/train/gpt5-chat-filtered-7b-adversarial-lr1e-6.sh \
  --exp_name gpt5-chat-filtered-7b-adversarial-lr1e-6 \
  --resume_step $STEP \
  --nnodes 1
```

## 🧪 Evaluation

To generate outputs for evaluation:

```bash
cd verl
git checkout eval
cd ..
bash scripts/generate/parallel_generate.sh
```

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{ye2025blackboxonpolicydistillationlarge,
  title={Black-Box On-Policy Distillation of Large Language Models},
  author={Tianzhu Ye and Li Dong and Zewen Chi and Xun Wu and Shaohan Huang and Furu Wei},
  journal={arXiv preprint arXiv:2511.10643},
  year={2025},
  url={https://arxiv.org/abs/2511.10643}
}
```

## 📧 Contact

For any questions or issues, please open an issue in this repository.