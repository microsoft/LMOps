# MiniLLM: Knowledge Distillation of Large Language Models

[paper](https://arxiv.org/abs/2306.08543) | [huggingface](https://huggingface.co/MiniLLM)

![Method](./figures/method.png)

![Results](./figures/results.png)

See also:
+ [DPKD](https://github.com/microsoft/LMOps/tree/main/dpkd): A simple improvement of MiniLLM using DPO.
+ [MiniPLM](https://github.com/thu-coai/MiniPLM): Knowledge distillation for **pre-training** lanuage models.

## 1 Environment
```bash
pip3 install git+https://github.com/t1101675/transformers@minillm
pip3 install torch
pip3 install deepspeed
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
pip3 install accelerate
pip3 install datasets
pip3 install peft
```
or
```bash
bash install.sh
```

Our data and pre-trained models are uploaded to our HuggingFace [repo](https://huggingface.co/MiniLLM).
We modified the [transformers code base](https://github.com/t1101675/transformers/tree/minillm) to support model (tensor) parallel and teacher-mixed sampling. The modified lines are wrapped with
```
# ### MiniLLM BEGIN ###
... SOME NEW CODES ...
# ### MiniLLM END ###
```

## 2 Data
### 2.1 Resources
+ The training/evaluation intruction-response data before processing can be downloaded from the following links: [dolly](https://huggingface.co/datasets/MiniLLM/dolly), [self-inst](https://huggingface.co/datasets/MiniLLM/self-inst), [vicuna](https://huggingface.co/datasets/MiniLLM/Vicuna), [sinst](https://huggingface.co/datasets/MiniLLM/sinst), and [uinst](https://huggingface.co/datasets/MiniLLM/uinst)
  ```bash
  huggingface-cli download MiniLLM/dolly --repo-type dataset /PATH_TO/LMOps/minillm/data/dolly/
  huggingface-cli download MiniLLM/self-inst --repo-type dataset /PATH_TO/LMOps/minillm/data/self-inst/
  huggingface-cli download MiniLLM/Vicuna --repo-type dataset /PATH_TO/LMOps/minillm/data/vicuna/
  huggingface-cli download MiniLLM/sinst --repo-type dataset /PATH_TO/LMOps/minillm/data/sinst/
  huggingface-cli download MiniLLM/uinst --repo-type dataset /PATH_TO/LMOps/minillm/data/uinst/
  ```
+ (Optional) The plain-text corpus $\mathcal{D}_\text{PT}$ can be download from the HugginFace datasets [repository](https://huggingface.co/datasets/openwebtext). For reproducibility, we recommend you to use the following preprocessed data.
+ The processed data can be downloaded from the following links: [dolly](https://huggingface.co/datasets/MiniLLM/dolly-processed), [openwebtext](https://huggingface.co/datasets/MiniLLM/openwebtext-processed) (Optional), [roberta-corpus](https://huggingface.co/datasets/MiniLLM/roberta-corpus-processed) (Optional).
  ```bash
  huggingface-cli download MiniLLM/dolly-processed --repo-type dataset --local-dir /PATH_TO/LMOps/minillm/processed_data/dolly/
  huggingface-cli download MiniLLM/openwebtext-processed --repo-type dataset --local-dir /PATH_TO/LMOps/minillm/processed_data/openwebtext/gpt2/512/10M/ # Optional
  huggingface-cli download MiniLLM/roberta-corpus-processed --repo-type dataset --local-dir /PATH_TO/LMOps/minillm/processed_data/openwebtext/ # Optional
  ```


### 2.2 Data Processing
#### SFT Data ($\mathcal{D}$ in paper)
```bash
bash scripts/gpt2/tools/process_data_dolly.sh /PATH_TO/LMOps/minillm # Process Dolly Train / Validation Data
bash scripts/opt/tools/process_data_dolly.sh /PATH_TO/LMOps/minillm # Process Dolly Train / Validation Data
bash scripts/llama/tools/process_data_dolly.sh /PATH_TO/LMOps/minillm # Process Dolly Train / Validation Data
```

#### (Optional) Plain-text Corpus ($\mathcal{D}_\text{PT}$ in paper)
Get plain-text corpus $\mathcal{D}_\text{PT}$:
```bash
python3 tools/get_openwebtext.py
```
This script will replace the continuous `\n` in each document with a special token "<@x(x!>" and write each document in OpenWebText in a line, which is covenient for parallel processing. In `data/openwebtext/data.txt`, we give an example of the resulting format. You can follow this format to prepare other corpus beyond OpenWebText.

Tokenize the data and store them in binary files:
```bash
bash scripts/gpt2/tools/process_data_pretrain.sh /PATH_TO/LMOps/minillm # Process OpenWebText Train / Validation Data
bash scripts/opt/tools/process_data_pretrain.sh /PATH_TO/LMOps/minillm # Process RoBERTa Corpus Train / Validation Data
bash scripts/llama/tools/process_data_pretrain.sh /PATH_TO/LMOps/minillm # Process RoBERTa Corpus Train / Validation Data
```

## 3 Models
### 3.1 Resources
+ The pre-trained models (MiniLLM and the baselines) can be found in this [collection](https://huggingface.co/collections/MiniLLM/minillm-66f51b3d667b4ee25046dafd).

#### Base Pre-trained Models
To run fine-tuning or standard KD baselines, you need to download the model checkpoints from [Huggingface Model Hub] and put them in `checkpoints/`. For example, for gpt2-large, you can download the model from this [link](https://huggingface.co/gpt2-large/tree/main) and put them in `checkpoints/gpt2-large`.
  ```bash
  huggingface-cli download gpt2 --repo-type model /PATH_TO/LMOps/minillm/checkpoints/gpt2-base
  huggingface-cli download gpt2-medium --repo-type model /PATH_TO/LMOps/minillm/checkpoints/gpt2-medium
  huggingface-cli download gpt2-large --repo-type model /PATH_TO/LMOps/minillm/checkpoints/gpt2-large
  huggingface-cli download gpt2-xl --repo-type model /PATH_TO/LMOps/minillm/checkpoints/gpt2-xlarge
  ```

Alternatively, you can also change the `CKPT` variable in each script to the corresponding model name to enable Transformers to download the base models automatically. For example, set `CKPT="gpt2-large"` in `scripts/gpt2/sft/sft_large.sh` causes download of the gpt2-large base model from the HugginFace model hub.

**NOTE:** 
1. LLaMA models require license and cannot be directly downloaded. 
2. If you want to use model parallel for training, it is recommended to download the models to `checkpoints` because you need to run `tools/convert_mp.py` to change their model parallel sizes (see next section).

### (Optional) 3.2 Change Model Parallel Size
If you find the model is too large to fit in your GPUs, you can increase/decrease the tensor parallel sizes with
```bash
python3 tools/convert_mp.py \
    --input_path results/llama/train/minillm/7B-init-13B-sft \
    --source_mp_size 1 \
    --target_mp_size 4 \
    --model_type llama # choose from opt and llama
```
To use the model with Model Parallel, we provide two example scripts for [training](https://github.com/microsoft/LMOps/tree/main/minillm/scripts/llama/sft/sft_7B_mp4.sh) and [evaluation](https://github.com/microsoft/LMOps/tree/main/minillm/scripts/llama/sft/eval_main_dolly_mp4.sh).

NOTE: Model parallelism is not applied to gpt2 because these models are generally sufficiant small to fit in common GPUs.

## 4 Run Evaluation
```bash
bash scripts/gpt2/eval/run_eval.sh /PATH_TO/LMOps/minillm
bash scripts/opt/eval/run_eval.sh /PATH_TO/LMOps/minillm
bash scripts/llama/eval/run_eval.sh /PATH_TO/LMOps/minillm
```

## 5 Train
We provide example commands for GPT-2 models. Similar scripts for model families can be found in `scripts/opt` and `scripts/llama`. All our experiments are conducted on 16 \* 32V100, which can be reduced for small models.
Some large models require tensor parallel size = 4, which is set in the scripts with `--model-parallel` and `--model-parallel-size` options.

### 5.1 Baselines
The final checkpoints are selected by the Rouge-L scores.
#### Fine-tune the teacher models
```bash
bash scripts/gpt2/sft/sft_xlarge.sh /PATH_TO/LMOps/minillm
```
Fine-tuned teacher model:
+ [teacher-gpt2-xlarge](https://huggingface.co/MiniLLM/teacher-gpt2-1.5B)

#### SFT Baselines
```bash
bash scripts/gpt2/sft/sft_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_medium.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_large.sh /PATH_TO/LMOps/minillm
```

The SFT models
+ [SFT-gpt2-base](https://huggingface.co/MiniLLM/SFT-gpt2-120M)
+ [SFT-gpt2-medium](https://huggingface.co/MiniLLM/SFT-gpt2-340M)
+ [SFT-gpt2-large](https://huggingface.co/MiniLLM/SFT-gpt2-760M)

#### KD Baselines
```bash
bash scripts/gpt2/kd/kd_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/kd/kd_medium.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/kd/kd_large.sh /PATH_TO/LMOps/minillm
```

The KD models
+ [KD-gpt2-base](https://huggingface.co/MiniLLM/KD-gpt2-120M)
+ [KD-gpt2-medium](https://huggingface.co/MiniLLM/KD-gpt2-340M)
+ [KD-gpt2-large](https://huggingface.co/MiniLLM/KD-gpt2-760M)

#### SeqKD Baselines
Generate and process responses with the teacher:
```bash
bash scripts/gpt2/tools/generate_data_seqkd.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/tools/process_pseudo_data_seqkd.sh /PATH_TO/LMOps/minillm
```
Fine-tune the model with SeqKD:
```bash
bash scripts/gpt2/seqkd/seqkd_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/seqkd/seqkd_medium.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/seqkd/seqkd_large.sh /PATH_TO/LMOps/minillm
```

The SeqKD models
+ [SeqKD-gpt2-base](https://huggingface.co/MiniLLM/SeqKD-gpt2-120M)
+ [SeqKD-gpt2-medium](https://huggingface.co/MiniLLM/SeqKD-gpt2-340M)
+ [SeqKD-gpt2-large](https://huggingface.co/MiniLLM/SeqKD-gpt2-760M)


### 5.2 MiniLLM
#### Initial Checkpoints
We first conduct SFT on base models to get a better initialization for the following RL-based MiniLLM training.

```bash
bash scripts/gpt2/sft/sft_base.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_medium.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/sft/sft_large.sh /PATH_TO/LMOps/minillm
```
The final checkpoints are selected by the **validation loss**. The trained checkpoints:
+ [init-gpt2-base](https://huggingface.co/MiniLLM/init-gpt2-120M)
+ [init-gpt2-medium](https://huggingface.co/MiniLLM/init-gpt2-340M)
+ [init-gpt2-large](https://huggingface.co/MiniLLM/init-gpt2-760M)

#### MiniLLM Training
The final checkpoints are selected by the Rouge-L scores.
```bash
bash scripts/gpt2/minillm/train_base_xl.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/minillm/train_medium_xl.sh /PATH_TO/LMOps/minillm
bash scripts/gpt2/minillm/train_large_xl.sh /PATH_TO/LMOps/minillm
```

For the data we use:
+ `PROMPT_DATA_DIR` is the SFT data ($\mathcal{D}$, Dolly), which is required.
+ `LM_DATA_DIR` is the plain-text corpus ($\mathcal{D}_\text{PT}$), which is optional. See `minillm/scripts/gpt2/minillm/train_base_xl_no_pt.sh` for training without `LM_DATA_DIR` (by just commenting out the `OPTS+=" --lm-data-dir ${LM_DATA_DIR}"` line).

The MiniLLM models
+ [MiniLLM-gpt2-base](https://huggingface.co/MiniLLM/MiniLLM-gpt2-120M)
+ [MiniLLM-gpt2-medium](https://huggingface.co/MiniLLM/MiniLLM-gpt2-340M)
+ [MiniLLM-gpt2-large](https://huggingface.co/MiniLLM/MiniLLM-gpt2-760M)


### 5.3 Multi-Node training
Multi-Node training is launched by `deepspeed`. We provide an example script in `scripts/llama/sft/sft_7B_mn.sh` for multi-node training. Compared to single-node scripts, some of the `DISTRIBUTED_ARGS` are changed, and you need to specify a hostfile like `configs/hostfiles/node_0_1` to tell the script which nodes to use. For more information, please refer to HuggingFace's [tutorial](https://huggingface.co/docs/transformers/main_classes/deepspeed#the-deepspeed-launcher).


## 6 Citation
```bibtex
@inproceedings{minillm,
  title={MiniLLM: Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  booktitle={Proceedings of ICLR},
  year={2024}
}
```
