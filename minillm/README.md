# MiniLLM: Knowledge Distillation of Large Language Models

![Method](./figures/method.png)
## 1 Setup
### 1.1 Environment
```bash
pip3 install -e transformers/
pip3 install deepspeed==0.8.0
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install nltk
pip3 install numerize
pip3 install rouge-score
pip3 install torchtyping
pip3 install rich
```
or
```bash
bash install.sh
```

### 1.2 Download
The resources of this repo can be download from this [link]().

## 2 Data
### 2.1 Resources
The training/evaluation data are in `data/`. The preprocessed data are in `processed_data/`.

### 2.2 Data Processing
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
The checkpoints are in `results/`.

### 3.2 Change Model Parallel Size
You can increase/decrease the tensor parallel sizes with
```bash
python3 tools/convert_mp.py \
    --input_path results/llama/train/llama-7B/minillm/ \
    --source_mp_size 1 \
    --target_mp_size 4 \
    --model_type llama # choose from opt and llama
```

## 4 Run Evaluation
```bash
bash scripts/gpt2/run.sh /PATH/TO/MiniLLM
bash scripts/opt/run.sh /PATH/TO/MiniLLM
bash scripts/llama/run.sh /PATH/TO/MiniLLM
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
Coming Soon