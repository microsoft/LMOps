# Evaluate [AdaptLLM](https://huggingface.co/AdaptLLM) and [Instruction-Pretrain](https://huggingface.co/instruction-pretrain) on domain-specific tasks

**NOTE: this script does NOT fit for the data format for chat models.**

## Setup
Ensure now you are in the `adaptllm` folder, if you have not installed the requirements:

```bash
pip install -r requirements.txt
```

## 1. Biomedicine Benchmarks
To reproduce the results of in the biomedicine domain:
```bash
DOMAIN='biomedicine'

# if the model can fit on a single GPU: set MODEL_PARALLEL=False
# elif the model is too large to fit on a single GPU: set MODEL_PARALLEL=True
MODEL_PARALLEL=False

# number of GPUs, chosen from [1,2,4,8]
N_GPU=8

# AdaptLLM-7B pre-trained from Llama1-7B
add_bos_token=False # this is set to False for AdaptLLM, and True for instruction-pretrain
bash scripts/inference.sh ${DOMAIN} 'AdaptLLM/medicine-LLM' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}

# AdaptLLM-13B pre-trained from Llama1-13B
add_bos_token=False
bash scripts/inference.sh ${DOMAIN} 'AdaptLLM/medicine-LLM-13B' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}

# medicine-Llama-8B pre-trained from Llama3-8B in Instruction Pretrain
add_bos_token=True
bash scripts/inference.sh ${DOMAIN} 'instruction-pretrain/medicine-Llama3-8B' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}
```

## 2. Finance Benchmarks
To reproduce the results in the finance domain:
```bash
DOMAIN='finance'

# if the model can fit on a single GPU: set MODEL_PARALLEL=False
# elif the model is too large to fit on a single GPU: set MODEL_PARALLEL=True
MODEL_PARALLEL=False

# number of GPUs, chosen from [1,2,4,8]
N_GPU=8

# AdaptLLM-7B pre-trained from Llama1-7B
add_bos_token=False # this is set to False for AdaptLLM, and True for instruction-pretrain
bash scripts/inference.sh ${DOMAIN} 'AdaptLLM/finance-LLM' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}

# finance-Llama-8B pre-trained from Llama3-8B in Instruction Pretrain
add_bos_token=True
bash scripts/inference.sh ${DOMAIN} 'instruction-pretrain/finance-Llama3-8B' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}
```

## 3. Law Benchmarks
To reproduce the results in the law domain:
```bash
DOMAIN='law'

# if the model can fit on a single GPU: set MODEL_PARALLEL=False
# elif the model is too large to fit on a single GPU: set MODEL_PARALLEL=True
MODEL_PARALLEL=False

# number of GPUs, chosen from [1,2,4,8]
N_GPU=8

# AdaptLLM-7B pre-trained from Llama1-7B
add_bos_token=False # this is set to False for AdaptLLM, and True for instruction-pretrain
bash scripts/inference.sh ${DOMAIN} 'AdaptLLM/law-LLM' ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}
```