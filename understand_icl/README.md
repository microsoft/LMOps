# Why Can GPT Learn In-Context?
This repository contains the implementation of ACL 2023 Findings paper "Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers".

## Installation
We recommend you to run the code using the docker under Linux:
```bash
docker run -it --rm--runtime=nvidia --ipc=host --privileged damaidai/icl:v1 bash
```
Then install the following packages with pip:
```bash
pip install --user datasets=2.4.0
pip install --user tensorboard scikit-learn
pip install --user jsonlines
pip install --user -e fairseq/
```

## Downloading Models
We use [Fairseq-LM](https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm) (dense models) in our experiments. You can download different sized model and the dictionary though the above link. For tokenizer, you can download [ENCODER](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json) and [BPE_VOCAB](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe).

## Analyzing Models

### Step 1: Producing the recording information
```bash
bash run.sh ${model_name} ${model_arch} ${task} ${k} ${seed} ${perm_id} ${output_path} ${base_dir} ${lr}
```
- `model_name`: Specify the model to analyze, e.g., en_dense_lm_1_3b.
- `model_arch`: Each model corresponds to a specific architecture: `gptmodel_large` (1.3B), `gptmodel_xl` (2.7B)
- `task`: Evaluation dataset. Could be ["cb", "sst2", "sst5", "subj", "mr", "agnews"].
- `k`: The number of demonstration examples (k-shot).
- `seed`: Random seed. Please refer to our paper to set the seed for each setting.
- `perm_id`: Permutation method for demonstration examples. In our experiments, it is always set to 0. 
- `output_path`: The output path.
- `base_dir`: Base directory for experiments. 
- `lr`: Learning rate used in the finetuning setting. Please refer to our paper to set it.

In `run.sh`, based on `base_dir`, 
- `bpe_path=$base_dir/gpt_icl/vocab.bpe`: The vocabulary file of BPE.
- `encoder_path=$base_dir/gpt_icl/encoder.json`: The encoder file of BPE.
- `dict_path=$base_dir/gpt_icl/$model_name/dict.txt`: The dictionary file.
- `ana_rlt_dir=$base_dir/ana_rlt/$model_name/$task`: The directory to save analysis results. 
- `model_path=$base_dir/gpt_icl/$model_name/model.pt`: Model file.
- `save_dir=$base_dir/ft_gpt/$task/$model_name/$lr`: Path to save the finetuned model in the finetuning setting. 

You should put the downloaded bpe file, encoder file, dictionary file, and model checkpoint in the corresponding path, and then run `run.sh`. When `run.sh` is done, `$ana_rlt_dir/$ana_setting/record_info.jsonl` will contain the recording information for analysis, where `$ana_setting` can be ["ftzs", "zs", "icl"]. They are corresponding to the finetuning, zero-shot, and ICL setting, respectively. 

### Step 2: Computing the analysis results
When the recording information for all models and tasks are produced, run `analyze.sh` to compute the analysis results in this paper. 
```bash
bash analyze.sh
```
The results will be saved at `$base_dir/ana_rlt/rlt_json/$task-$model.json` or `$base_dir/ana_rlt/rlt_json/$task-$model_training_attn.json`

## Citation
Coming soon.