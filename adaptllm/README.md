# Adapting Large Language Models to Domains

This repo contains the model, code and data for our paper [Adapting Large Language Models via Reading Comprehension](https://huggingface.co/papers/2309.09530)

We explore **continued pre-training on domain-specific corpora** for large language models. While this approach enriches LLMs with domain knowledge, it significantly hurts their prompting ability for question answering. Inspired by human learning via reading comprehension, we propose a simple method to **transform large-scale pre-training corpora into reading comprehension texts**, consistently improving prompting performance across tasks in **biomedicine, finance, and law domains**. Our 7B model competes with much larger domain-specific models like BloombergGPT-50B.

### 🤗 [2024/6/21] We release the 2nd version of AdaptLLM at [Instruction-Pretrain](https://huggingface.co/instruction-pretrain) 🤗

**************************** **Updates** ****************************
* 2024/6/22: Released the [benchmarking code](https://github.com/microsoft/LMOps/tree/main/adaptllm).
* 2024/6/21: Released the 2nd version of AdaptLLM at [Instruction-Pretrain](https://huggingface.co/instruction-pretrain).
* 2024/1/16: Our [research paper](https://huggingface.co/papers/2309.09530) has been accepted by ICLR 2024.
* 2023/12/19: Released our [13B base models](https://huggingface.co/AdaptLLM/law-LLM-13B) developed from LLaMA-1-13B.
* 2023/12/8: Released our [chat models](https://huggingface.co/AdaptLLM/law-chat) developed from LLaMA-2-Chat-7B.
* 2023/9/18: Released our [paper](https://huggingface.co/papers/2309.09530), [code](https://github.com/microsoft/LMOps), [data](https://huggingface.co/datasets/AdaptLLM/law-tasks), and [base models](https://huggingface.co/AdaptLLM/law-LLM) developed from LLaMA-1-7B.


# Domain-specific LLMs
Our models of different domains are now available in Huggingface: [biomedicine-LLM](https://huggingface.co/AdaptLLM/medicine-LLM), [finance-LLM](https://huggingface.co/AdaptLLM/finance-LLM) and [law-LLM](https://huggingface.co/AdaptLLM/law-LLM), the performances of our AdaLLM compared to other domain-specific LLMs are:

<p align='center'>
    <img src="./images/comparison.png" width="700">
</p>

We also scale up the model size to 13B, and train from chat models: 
* Scale up to 13B: [Biomedicine-LLM-13B](https://huggingface.co/AdaptLLM/medicine-LLM-13B), [Finance-LLM-13B](https://huggingface.co/AdaptLLM/finance-LLM-13B) and [Law-LLM-13B](https://huggingface.co/AdaptLLM/law-LLM-13B)
* Chat models: [Biomedicine-Chat](https://huggingface.co/AdaptLLM/medicine-chat), [Finance-Chat](https://huggingface.co/AdaptLLM/finance-chat) and [Law-Chat](https://huggingface.co/AdaptLLM/law-chat)

# Domain-specific Tasks
To easily reproduce our results, we have uploaded the filled-in zero/few-shot input instructions and output completions of each domain-specific task: [biomedicine-tasks](https://huggingface.co/datasets/AdaptLLM/medicine-tasks), [finance-tasks](https://huggingface.co/datasets/AdaptLLM/finance-tasks), and [law-tasks](https://huggingface.co/datasets/AdaptLLM/law-tasks).

# Data processing and benchmarking code
## Install Dependencies
```bash
pip install -r requirements.txt
```

## Data: Transfer Raw Corpora into Reading Comprehension
Our method is very **simple**, highly **scalable** and **applicable** to any pre-training corpora.

Try transferring the raw texts in the [data_samples](./data_samples/README.md) folder:
```
python raw2read.py
```

## Benchmark: Evaluate Our models on Domain-specific Tasks
To evaluate our models on the domain-specific tasks:
```bash
# domain name, chosen from ['biomedicine', 'finance', 'law']
DOMAIN='biomedicine'

# hf model names chosen from the following (NOT applicable to chat models):
# ['AdaptLLM/medicine-LLM', 'AdaptLLM/finance-LLM', 'AdaptLLM/law-LLM', 
#  'AdaptLLM/medicine-LLM-13B', 'AdaptLLM/finance-LLM-13B', 'AdaptLLM/law-LLM-13B',
#  'instruction-pretrain/medicine-Llama3-8B', instruction-pretrain/finance-Llama3-8B]
MODEL='instruction-pretrain/medicine-Llama3-8B'

# if the model can fit on a single GPU: set MODEL_PARALLEL=False
# elif the model is too large to fit on a single GPU: set MODEL_PARALLEL=True
MODEL_PARALLEL=False

# number of GPUs, chosen from [1,2,4,8]
N_GPU=8

# whether to add_bos_token, this is set to False for AdaptLLM, and True for instruction-pretrain
add_bos_token=True

bash scripts/inference.sh ${DOMAIN} ${MODEL} ${add_bos_token} ${MODEL_PARALLEL} ${N_GPU}
```
We include detailed instructions [here](./scripts/README.md)

## Citation
```bibtex
@inproceedings{
cheng2024adapting,
title={Adapting Large Language Models via Reading Comprehension},
author={Daixuan Cheng and Shaohan Huang and Furu Wei},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=y886UXPEZ0}
}
```