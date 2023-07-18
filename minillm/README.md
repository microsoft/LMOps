# MiniLLM: Knowledge Distillation of Large Language Models

![Method](./figures/method.png)
## 1 Environment
```bash
pip3 install -e transformers/
pip3 install deepspeed==0.8.0
pip3 install pydantic==1.10.8
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install nltk
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
pip3 install datasets
```
or
```bash
bash install.sh
```

## 2 Data
### 2.1 Resources
+ The training/evaluation intruction-response data before processing can be downloaded from this [link](https://conversationhub.blob.core.windows.net/beit-share-public/MiniLLM/data.tar?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D).
+ The plain-text corpus $\mathcal{D}_\text{PT}$ can be download from the HugginFace datasets [repository](https://huggingface.co/datasets/openwebtext). For reproducibility, we recommend you to use the following preprocessed data.
+ The processed data can be downloaded from this [link](https://conversationhub.blob.core.windows.net/beit-share-public/MiniLLM/processed_data.tar?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D).


### 2.2 Data Processing
Get plain-text corpus $\mathcal{D}_\text{PT}$:
```bash
python3 tools/get_openwebtext.py
```
This script will replace the continuous `\n` in each document with a special token "<@x(x!>" and write each document in OpenWebText in a line, which is covenient for parallel processing. In `data/openwebtext/data.txt`, we give an example of the resulting format. You can follow this format to prepare other corpus beyond OpenWebText.

Tokenize the data and store them in binary files:
```bash
bash scripts/gpt2/tools/process_data_dolly.sh /PATH/TO/MiniLLM # Process Dolly Train / Validation Data
bash scripts/gpt2/tools/process_data_pretrain.sh /PATH/TO/MiniLLM # Process OpenWebText Train / Validation Data

bash scripts/opt/tools/process_data_dolly.sh /PATH/TO/MiniLLM # Process Dolly Train / Validation Data
bash scripts/opt/tools/process_data_pretrain.sh /PATH/TO/MiniLLM # Process RoBERTa Corpus Train / Validation Data

bash scripts/llama/tools/process_data_dolly.sh /PATH/TO/MiniLLM # Process Dolly Train / Validation Data
bash scripts/llama/tools/process_data_pretrain.sh /PATH/TO/MiniLLM # Process RoBERTa Corpus Train / Validation Data
```

## 3 Models
### 3.1 Resources
+ The baselines and MiniLLM models based on GPT-2 can be downloaded from this [link](https://conversationhub.blob.core.windows.net/beit-share-public/MiniLLM/gpt2.tar?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D).
+ The baselines and MiniLLM models based on OPT can be downloaded from this [link](https://conversationhub.blob.core.windows.net/beit-share-public/MiniLLM/opt.tar?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D).
+ The baselines and MiniLLM models based on LLaMA can be downloaded from this [link](https://conversationhub.blob.core.windows.net/beit-share-public/MiniLLM/llama.tar?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D).

### 3.2 Change Model Parallel Size
You can increase/decrease the tensor parallel sizes with
```bash
python3 tools/convert_mp.py \
    --input_path results/llama/train/minillm/7B-init-13B-sft \
    --source_mp_size 1 \
    --target_mp_size 4 \
    --model_type llama # choose from opt and llama
```

## 4 Run Evaluation
```bash
bash scripts/gpt2/eval/run_eval.sh /PATH/TO/MiniLLM
bash scripts/opt/eval/run_eval.sh /PATH/TO/MiniLLM
bash scripts/llama/eval/run_eval.sh /PATH/TO/MiniLLM
```

## 5 Train
We provide example commands for GPT-2 models. Similar scripts for model families can be found in `scripts/opt` and `scripts/llama`. All our experiments are conducted on 16 \* 32V100, which can be reduced for small models.
Some large models require tensor parallel size = 4, which is set in the scripts with `--model-parallel` and `--model-parallel-size` options.

### 5.1 Baselines
The final checkpoints are selected by the Rouge-L scores.
#### Fine-tune the teacher models
```bash
bash scripts/gpt2/sft/sft_xlarge.sh /PATH/TO/MiniLLM
```
#### SFT Baselines
```bash
bash scripts/gpt2/sft/sft_base.sh /PATH/TO/MiniLLM
bash scripts/gpt2/sft/sft_medium.sh /PATH/TO/MiniLLM
bash scripts/gpt2/sft/sft_large.sh /PATH/TO/MiniLLM
```

#### KD Baselines
```bash
bash scripts/gpt2/kd/kd_base.sh /PATH/TO/MiniLLM
bash scripts/gpt2/kd/kd_medium.sh /PATH/TO/MiniLLM
bash scripts/gpt2/kd/kd_large.sh /PATH/TO/MiniLLM
```

#### SeqKD Baselines
Generate and process responses with the teacher:
```bash
bash scripts/gpt2/tools/generate_data_seqkd.sh /PATH/TO/MiniLLM
bash scripts/gpt2/tools/process_pseudo_data_seqkd.sh /PATH/TO/MiniLLM
```
Fine-tune the model with SeqKD:
```bash
bash scripts/gpt2/seqkd/seqkd_base.sh /PATH/TO/MiniLLM
bash scripts/gpt2/seqkd/seqkd_medium.sh /PATH/TO/MiniLLM
bash scripts/gpt2/seqkd/seqkd_large.sh /PATH/TO/MiniLLM
```

### 5.2 MiniLLM
#### Initial Checkpoints
The final checkpoints are selected by the **validation loss**.
```bash
bash scripts/gpt2/sft/sft_base.sh /PATH/TO/MiniLLM
bash scripts/gpt2/sft/sft_medium.sh /PATH/TO/MiniLLM
bash scripts/gpt2/sft/sft_large.sh /PATH/TO/MiniLLM
```

#### Train
The final checkpoints are selected by the Rouge-L scores.
```bash
bash scripts/gpt2/minillm/train_base_xl.sh /PATH/TO/MiniLLM
bash scripts/gpt2/minillm/train_medium_xl.sh /PATH/TO/MiniLLM
bash scripts/gpt2/minillm/train_large_xl.sh /PATH/TO/MiniLLM
```

## 6 Citation
```bibtex
@article{minillm,
  title={Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  journal={arXiv preprint arXiv:2306.08543},
  year={2023}
}
```