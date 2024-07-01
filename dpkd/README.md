# Direct Preference Knowledge Distillation for Large Language Models

## Environment
```bash
conda create -n dpkd python=3.11
conda activate dpkd
```
and

```bash
bash install.sh
```

## Run Distillation

Train runner:
```bash
bash scripts/dpkd-gpt2_base_runner.sh  PATH_TO_DPKD 
```

## Evaluation

Evaluation runner:
```bash
bash scripts/dpkd-gpt2_base_evaluate.sh  PATH_TO_DPKD 
```

## Citation

```bibtex
@article{dpkd,
  title={Direct Preference Knowledge Distillation for Large Language Models},
  author={Yixing Li and Yuxian Gu and Li Dong and Dequan Wang and Yu Cheng and Furu Wei},
  journal={arXiv preprint arXiv:2406.19774},
  year={2024}
}
```

