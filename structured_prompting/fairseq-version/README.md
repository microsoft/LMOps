# Structured Prompting
This repository contains the implementation of structured prompting.

## Installation
We recommend you to run the code using the docker under Linux:
```bash
docker run -it --rm --runtime=nvidia --ipc=host --privileged pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel bash
```
Then install the following packages with pip:
```bash
pip install --user datasets==2.4.0
pip install --user numpy==1.22
pip install --user tensorboard scikit-learn
pip install --user -e fairseq/
```

## Downloading Models
We use [Fairseq-LM](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm) (dense models) in our experiments. You can download different sized model and the dictionary though the above link. For tokenizer, you can download [ENCODER](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json) and [BPE_VOCAB](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe).

## Running Structured Prompting
```bash
sh scripts/manyshot.sh ${seed} ${task} ${model_path} ${model_arch} ${k} ${chunk_len} ${batch_size} ${ngpu} ${bpe_path} ${encoder_path} ${dict_path} ${output_path}
```
- `seed`: Use different seeds to use different demonstration examples. In our experiments, we use six random seeds (2,4,6,8,10,12) for each dataset and report the average performance.
- `task`: Evaluation dataset. We list all possible dataset names in `task_map` in `struprompting/tasks/large_icl_eval.py`. For StoryCloze, you need to download it manually.
- `model_path`: Model file.
- `model_arch`: Each model corresponds to a specific architecture: `gptmodel_large` (1.3B), `gptmodel_xl` (2.7B), `gptmodel_xxl` (6.7B), `gptmodel_huge` (13B).
- `k`: The number of demonstration examples (k-shot).
- `chunk_len`: The length of individual demonstration group. Default: 2000.
- `batch_size`: Batch size for inference. You need to modify the batch size of encoding demonstrations in the bash file.
- `ngpu`: The number of available GPUs. We support single-node evaluation with multiple GPUs.
- `bpe_path`: The vocabulary file of BPE.
- `encoder_path`: The encoder file of BPE.
- `dict_path`: The dictionary file.
- `output_path`: The output path.
  
The default alignment strategy for different groups is to truncate all of them to a fixed length (`chunk_len`) with the argument `--truncate`. You can try space padding by using `--pad-space` instead. If you disable both of them, the strategy is attention mask.

## Running Vanilla In-Context Learning
```bash
sh scripts/manyshot_baseline.sh ${seed} ${task} ${model_path} ${model_arch} ${k} ${batch_size} ${ngpu} ${bpe_path} ${encoder_path} ${dict_path} ${output_path}
```
The arguments are used in the same way as above. For vanilla in-context learning, `k` is limited by the context window size (2048).

## Citation

```
@inproceedings{structprompt,
  title={Structured Prompting: Scaling In-Context Learning to 1,000 Examples},
  author={Yaru Hao and Yutao Sun and Li Dong and Zhixiong Han and Yuxian Gu and Furu Wei},
  year={2022}
}
```
