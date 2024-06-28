# Direct Preference Knowledge Distillation for Large Language Models

## 1 Environment
```bash
conda create -n dpkd python=3.11
conda activate dpkd
```
and

```bash
bash install.sh
```


## 2 Runner
Train runner:
```bash
bash scripts/dpkd-gpt2_base_runner.sh  PATH_TO_DPKD 
```
Evaluation runner:
```bash
bash scripts/dpkd-gpt2_base_evaluate.sh  PATH_TO_DPKD 
```


## 3 Citation


