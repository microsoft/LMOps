# Black-Box On-Policy Distillation of Large Language Models

This repository contains the implementation and resources for our paper **"Black-Box On-Policy Distillation of Large Language Models"**.

📄 **Paper**: [arXiv:2511.10643](https://arxiv.org/abs/2511.10643)

💾 **Data**: [LMSYS-Chat-GPT-5-Chat-Response](https://huggingface.co/datasets/ytz20/LMSYS-Chat-GPT-5-Chat-Response)

🤖 **Models**: [GAD Models](https://huggingface.co/collections/ytz20/gad-models)

## 🚀 Getting Started 

### Environment Setup

We use `czwin32768/verl2:v0.2.0-vllm085` which has `python==3.10.12, pytorch==2.6.0, vllm==0.8.5` as the recommended docker image in the code snippet below. Note that the docker is not related to [VeRL](https://github.com/volcengine/verl); you can also setup a similar environment by your own. 

We use two repos as to easily install different branches for different experiments. Check this repo for environment setup and scripts for running experiments. Check algorithm implementation at `https://github.com/YTianZHU/verl.git` and it is installed in the code snippet below. The algorithm implementation repo is based on [VeRL](https://github.com/volcengine/verl). **We hack the `critic` module to use it as our discriminator**.

```bash
bash local_docker.sh
cd /tmp
git clone https://github.com/microsoft/LMOps.git
cd /tmp/LMOps/gad
git clone https://github.com/YTianZHU/verl.git
bash local_setup.sh
```

### Data Preparation

We provide teacher data from GPT-5-Chat at [Teacher Data](https://huggingface.co/datasets/ytz20/LMSYS-Chat-GPT-5-Chat-Response). Use code below to download and prepare data. 

```
python tools/export_lmsys_parquet.py
```

## 📦 Training

There are four branches in the installed GAD VeRL implementation repo: `seqkd` branch for running the SeqKD baseline, `warmup` branch for warmup stage of our method, `gad` branch for GAD training stage of our method and `eval` branch to use the already-trained model to perform generation only. We checkout to the corresponding branch before each experiment as shown in the scripts below.

For SeqKD and warmup stage of GAD, the student is supervised-finetuned on the teacher response (corresponding code at [sft_seqkd](https://github.com/YTianZHU/verl/blob/seqkd/verl/workers/actor/dp_actor.py#L485) and [sft_warmup](https://github.com/YTianZHU/verl/blob/warmup/verl/workers/actor/dp_actor.py#L495)). We choose to use this VeRL-based repo to implement them for best alignment.

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

Then we use GPT-4o to generate reference answer and perform automatic score evaluation. **You can also use open-source models (like Qwen2.5-72B-Instruct) to generate reference answer and to score the outputs.**

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