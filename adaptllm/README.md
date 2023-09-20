# Adapting Large Language Models via Reading Comprehension

This repo contains the model, code and data for our paper [Adapting Large Language Models via Reading Comprehension](https://arxiv.org/pdf/2309.09530.pdf)

We explore **continued pre-training on domain-specific corpora** for large language models. While this approach enriches LLMs with domain knowledge, it significantly hurts their prompting ability for question answering. Inspired by human learning via reading comprehension, we propose a simple method to **transform large-scale pre-training corpora into reading comprehension texts**, consistently improving prompting performance across tasks in **biomedicine, finance, and law domains**. Our 7B model competes with much larger domain-specific models like BloombergGPT-50B. Moreover, our domain-specific reading comprehension texts enhance model performance even on general benchmarks, indicating potential for developing a general LLM across more domains.


## Domain-specific LLMs:
Our models of different domains are now available in Huggingface: [Biomedicine-LLM](https://huggingface.co/AdaptLLM/medicine-LLM), [Finance-LLM](https://huggingface.co/AdaptLLM/finance-LLM) and [Law-LLM](https://huggingface.co/AdaptLLM/law-LLM), the performances of our AdaptLLM compared to other domain-specific LLMs are:

<p align='center'>
    <img src="./comparison.png" width="700">
</p>

## Domain-specific Tasks:
To easily reproduce our results, we have uploaded the filled-in zero/few-shot input instructions and output completions of each domain-specific task: [biomedicine-tasks](https://huggingface.co/datasets/AdaptLLM/medicine-tasks), [finance-tasks](https://huggingface.co/datasets/AdaptLLM/finance-tasks), and [law-tasks](https://huggingface.co/datasets/AdaptLLM/law-tasks).

## Code for Transferring Raw Corpora into Reading Comprehension Texts
Coming soon...

## Citation:
```bibtex
@inproceedings{AdaptLLM,
    title={Adapting Large Language Models via Reading Comprehension}, 
    author={Daixuan Cheng and Shaohan Huang and Furu Wei},
    url={https://arxiv.org/abs/2309.09530},
    year={2023},
}
```

