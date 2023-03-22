# Dense Passage Retrieval

Dense Passage Retrieval (`DPR`) - is a set of tools and models for state-of-the-art open-domain Q&A research.
It is based on the following paper:

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih. [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/abs/2004.04906) Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769–6781, 2020.

If you find this work useful, please cite the following paper:

```
@inproceedings{karpukhin-etal-2020-dense,
    title = "Dense Passage Retrieval for Open-Domain Question Answering",
    author = "Karpukhin, Vladimir and Oguz, Barlas and Min, Sewon and Lewis, Patrick and Wu, Ledell and Edunov, Sergey and Chen, Danqi and Yih, Wen-tau",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.550",
    doi = "10.18653/v1/2020.emnlp-main.550",
    pages = "6769--6781",
}
```

If you're interesting in reproducing experimental results in the paper based on our model checkpoints (i.e., don't want to train the encoders from scratch), you might consider using the [Pyserini toolkit](https://github.com/castorini/pyserini/blob/master/docs/experiments-dpr.md), which has the experiments nicely packaged in via `pip`.
Their toolkit also reports higher BM25 and hybrid scores.

## Features
1. Dense retriever model is based on bi-encoder architecture.
2. Extractive Q&A reader&ranker joint model inspired by [this](https://arxiv.org/abs/1911.03868) paper.
3. Related data pre- and post- processing tools.
4. Dense retriever component for inference time logic is based on FAISS index.

## New (March 2021) release
DPR codeabse is upgraded with a number of enhancements and new models.
Major changes:
1. [Hydra](https://hydra.cc/)-based configuration for all the command line tools exept the data loader (to be converted soon)
2. Pluggable data processing layer to support custom datasets
3. New retrieval model checkpoint with better perfromance.

## New (March 2021) retrieval model
A new bi-encoder model trained on NQ dataset only is now provided: a new checkpoint, training data, retrieval results and wikipedia embeddings.
It is trained on the original DPR NQ train set and its version where hard negatives are mined using DPR index itself using the previous NQ checkpoint.
A Bi-encoder model is trained from scratch using this new training data combined with our original NQ training data. This training scheme gives a nice retrieval performance boost.

New vs old top-k documents retrieval accuracy on NQ test set (3610 questions).

| Top-k passages        | Original DPR NQ model           | New DPR model  |
| ------------- |:-------------:| -----:|
| 1      | 45.87 | 52.47 |
| 5      | 68.14      |   72.24 |
| 20  | 79.97      |    81.33 |
| 100  | 85.87      |    87.29 |

New model downloadable resources names (see how to use download_data script below):

Checkpoint: checkpoint.retriever.single-adv-hn.nq.bert-base-encoder

New training data: data.retriever.nq-adv-hn-train

Retriever resutls for NQ test set: data.retriever_results.nq.single-adv-hn.test

Wikipedia embeddings: data.retriever_results.nq.single-adv-hn.wikipedia_passages


## Installation

Installation from the source. Python's virtual or Conda environments are recommended.

```bash
git clone git@github.com:facebookresearch/DPR.git
cd DPR
pip install .
```

DPR is tested on Python 3.6+ and PyTorch 1.2.0+.
DPR relies on third-party libraries for encoder code implementations.
It currently supports Huggingface (version <=3.1.0) BERT, Pytext BERT and Fairseq RoBERTa encoder models.
Due to generality of the tokenization process, DPR uses Huggingface tokenizers as of now. So Huggingface is the only required dependency, Pytext & Fairseq are optional.
Install them separately if you want to use those encoders.


## Resources & Data formats
First, you need to prepare data for either retriever or reader training.
Each of the DPR components has its own input/output data formats. 
You can see format descriptions below.
DPR provides NQ & Trivia preprocessed datasets (and model checkpoints) to be downloaded from the cloud using our dpr/data/download_data.py tool. One needs to specify the resource name to be downloaded. Run 'python data/download_data.py' to see all options.

```bash
python data/download_data.py \
	--resource {key from download_data.py's RESOURCES_MAP}  \
	[optional --output_dir {your location}]
```
The resource name matching is prefix-based. So if you need to download all data resources, just use --resource data.

## Retriever input data format
The default data format of the Retriever training data is JSON.
It contains pools of 2 types of negative passages per question, as well as positive passages and some additional information.

```
[
  {
	"question": "....",
	"answers": ["...", "...", "..."],
	"positive_ctxs": [{
		"title": "...",
		"text": "...."
	}],
	"negative_ctxs": ["..."],
	"hard_negative_ctxs": ["..."]
  },
  ...
]
```

Elements' structure  for negative_ctxs & hard_negative_ctxs is exactly the same as for positive_ctxs.
The preprocessed data available for downloading also contains some extra attributes which may be useful for model modifications (like bm25 scores per passage). Still, they are not currently in use by DPR.

You can download prepared NQ dataset used in the paper by using 'data.retriever.nq' key prefix. Only dev & train subsets are available in this format.
We also provide question & answers only CSV data files for all train/dev/test splits. Those are used for the model evaluation since our NQ preprocessing step looses a part of original samples set.
Use 'data.retriever.qas.*' resource keys to get respective sets for evaluation.

```bash
python data/download_data.py
	--resource data.retriever
	[optional --output_dir {your location}]
```

## DPR data formats and custom processing 
One can use their own data format and custom data parsing & loading logic by inherting from DPR's Dataset classes in dpr/data/{biencoder|retriever|reader}_data.py files and implementing load_data() and __getitem__() methods. See [DPR hydra configuration](https://github.com/facebookresearch/DPR/blob/master/conf/README.md) instructions.


## Retriever training
Retriever training quality depends on its effective batch size. The one reported in the paper used 8 x 32GB GPUs.
In order to start training on one machine:
```bash
python train_dense_encoder.py \
train_datasets=[list of train datasets, comma separated without spaces] \
dev_datasets=[list of dev datasets, comma separated without spaces] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```

Example for NQ dataset

```bash
python train_dense_encoder.py \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_local \
output_dir={path to checkpoints dir}
```

DPR uses HuggingFace BERT-base as the encoder by default. Other ready options include Fairseq's ROBERTA and Pytext BERT models.
One can select them by either changing encoder configuration files (conf/encoder/hf_bert.yaml) or providing a new configuration file in conf/encoder dir and enabling it with encoder={new file name} command line parameter. 

Notes:
- If you want to use pytext bert or fairseq roberta, you will need to download pre-trained weights and specify encoder.pretrained_file parameter. Specify the dir location of the downloaded files for 'pretrained.fairseq.roberta-base' resource prefix for RoBERTa model or the file path for pytext BERT (resource name 'pretrained.pytext.bert-base.model').
- Validation and checkpoint saving happens according to train.eval_per_epoch parameter value.
- There is no stop condition besides a specified amount of epochs to train (train.num_train_epochs configuration parameter).
- Every evaluation saves a model checkpoint.
- The best checkpoint is logged in the train process output.
- Regular NLL classification loss validation for bi-encoder training can be replaced with average rank evaluation. It aggregates passage and question vectors from the input data passages pools, does large similarity matrix calculation for those representations and then averages the rank of the gold passage for each question. We found this metric more correlating with the final retrieval performance vs nll classification loss. Note however that this average rank validation works differently in DistributedDataParallel vs DataParallel PyTorch modes. See train.val_av_rank_* set of parameters to enable this mode and modify its settings.

See the section 'Best hyperparameter settings' below as e2e example for our best setups.

## Retriever inference

Generating representation vectors for the static documents dataset is a highly parallelizable process which can take up to a few days if computed on a single GPU. You might want to use multiple available GPU servers by running the script on each of them independently and specifying their own shards.

```bash
python generate_dense_embeddings.py \
	model_file={path to biencoder checkpoint} \
	ctx_src={name of the passages resource, set to dpr_wiki to use our original wikipedia split} \
	shard_id={shard_num, 0-based} num_shards={total number of shards} \
	out_file={result files location + name PREFX}	
```

The name of the resource for ctx_src parameter 
or just the source name from conf/ctx_sources/default_sources.yaml file.

Note: you can use much large batch size here compared to training mode. For example, setting batch_size 128 for 2 GPU(16gb) server should work fine.
You can download already generated wikipedia embeddings from our original model (trained on NQ dataset) using resource key 'data.retriever_results.nq.single.wikipedia_passages'. 
Embeddings resource name for the new better model 'data.retriever_results.nq.single-adv-hn.wikipedia_passages'

We generally use the following params on 50 2-gpu nodes: batch_size=128 shard_id=0 num_shards=50



## Retriever validation against the entire set of documents:

```bash

python dense_retriever.py \
	model_file={path to a checkpoint downloaded from our download_data.py as 'checkpoint.retriever.single.nq.bert-base-encoder'} \
	qa_dataset={the name os the test source} \
	ctx_datatsets=[{list of passage sources's names, comma separated without spaces}] \
	encoded_ctx_files=[{list of encoded document files glob expression, comma separated without spaces}] \
	out_file={path to output json file with results} 
	
```

For example, If your generated embeddings fpr two passages set as ~/myproject/embeddings_passages1/wiki_passages_* and ~/myproject/embeddings_passages2/wiki_passages_* files and want to evaluate on NQ dataset:

```bash
python dense_retriever.py \
	model_file={path to a checkpoint file} \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki] \
	encoded_ctx_files=[\"~/myproject/embeddings_passages1/wiki_passages_*\",\"~/myproject/embeddings_passages2/wiki_passages_*\"] \
	out_file={path to output json file with results} 
```


The tool writes retrieved results for subsequent reader model training into specified out_file.
It is a json with the following format:

```
[
    {
        "question": "...",
        "answers": ["...", "...", ... ],
        "ctxs": [
            {
                "id": "...", # passage id from database tsv file
                "title": "",
                "text": "....",
                "score": "...",  # retriever score
                "has_answer": true|false
     },
]
```
Results are sorted by their similarity score, from most relevant to least relevant.

By default, dense_retriever uses exhaustive search process, but you can opt in to use lossy index types.
We provide HNSW and HNSW_SQ index options.
Enabled them by indexer=hnsw or indexer=hnsw_sq command line arguments.
Note that using this index may be useless from the research point of view since their fast retrieval process comes at the cost of much longer indexing time and higher RAM usage.
The similarity score provided is the dot product for the default case of exhaustive search (indexer=flat) and L2 distance in a modified representations space in case of HNSW index.


## Reader model training
```bash
python train_extractive_reader.py \
	encoder.sequence_length=350 \
	train_files={path to the retriever train set results file} \
	dev_files={path to the retriever dev set results file}  \
	output_dir={path to output dir}
```
Default hyperparameters are set for a single node with 8 gpus setup.
Modify them as needed in the conf/train/extractive_reader_default.yaml and conf/extractive_reader_train_cfg.yaml cpnfiguration files or override specific parameters from the command line.
First time run will preprocess train_files & dev_files and convert them into serialized set of .pkl files in the same locaion and will use them on all subsequent runs.

Notes:
- If you want to use pytext bert or fairseq roberta, you will need to download pre-trained weights and specify encoder.pretrained_file parameter. Specify the dir location of the downloaded files for 'pretrained.fairseq.roberta-base' resource prefix for RoBERTa model or the file path for pytext BERT (resource name 'pretrained.pytext.bert-base.model').
- Reader training pipeline does model validation every train.eval_step batches
- Like the bi-encoder, it saves model checkpoints on every validation
- Like the bi-encoder, there is no stop condition besides a specified amount of epochs to train.
- Like the bi-encoder, there is no best checkpoint selection logic, so one needs to select that based on dev set validation performance which is logged in the train process output.
- Our current code only calculates the Exact Match metric.

## Reader model inference

In order to make an inference, run `train_reader.py` without specifying `train_files`. Make sure to specify `model_file` with the path to the checkpoint, `passages_per_question_predict` with number of passages per question (being used when saving the prediction file), and `eval_top_docs` with a list of top passages threshold values from which to choose question's answer span (to be printed as logs). The example command line is as follows.

```bash
python train_extractive_reader.py \
  prediction_results_file={path to a file to write the results to} \
  eval_top_docs=[10,20,40,50,80,100] \
  dev_files={path to the retriever results file to evaluate} \
  model_file= {path to the reader checkpoint} \
  train.dev_batch_size=80 \
  passages_per_question_predict=100 \
  encoder.sequence_length=350
```

## Distributed training
Use Pytorch's distributed training launcher tool:

```bash
python -m torch.distributed.launch \
	--nproc_per_node={WORLD_SIZE}  {non distributed scipt name & parameters}
```
Note:
- all batch size related parameters are specified per gpu in distributed mode(DistributedDataParallel) and for all available gpus in DataParallel (single node - multi gpu) mode.

## Best hyperparameter settings

e2e example with the best settings for NQ dataset.

### 1. Download all retriever training and validation data:

```bash
python data/download_data.py --resource data.wikipedia_split.psgs_w100
python data/download_data.py --resource data.retriever.nq
python data/download_data.py --resource data.retriever.qas.nq
```

### 2. Biencoder(Retriever) training in the single set mode.

We used distributed training mode on a single 8 GPU x 32 GB server

```bash
python -m torch.distributed.launch --nproc_per_node=8
train_dense_encoder.py \
train=biencoder_nq \
train_datasets=[nq_train] \
dev_datasets=[nq_dev] \
train=biencoder_nq \
output_dir={your output dir}
```

New model training combines two NQ datatsets:

```bash
python -m torch.distributed.launch --nproc_per_node=8
train_dense_encoder.py \
train=biencoder_nq \
train_datasets=[nq_train,nq_train_hn1] \
dev_datasets=[nq_dev] \
train=biencoder_nq \
output_dir={your output dir}
```

This takes about a day to complete the training for 40 epochs. It switches to Average Rank validation on epoch 30 and it should be around 25 or less at the end.
The best checkpoint for bi-encoder is usually the last, but it should not be so different if you take any after epoch ~ 25.

### 3. Generate embeddings for Wikipedia.
Just use instructions for "Generating representations for large documents set". It takes about 40 minutes to produce 21 mln passages representation vectors on 50 2 GPU servers.

### 4. Evaluate retrieval accuracy and generate top passage results for each of the train/dev/test datasets.

```bash

python dense_retriever.py \
	model_file={path to the best checkpoint or use our proivded checkpoints (Resource names like checkpoint.retriever.*)  } \
	qa_dataset=nq_test \
	ctx_datatsets=[dpr_wiki] \
	encoded_ctx_files=["{glob expression for generated embedding files}"] \
	out_file={path to the output file}
```

Adjust batch_size based on the available number of GPUs, 64-128 should work for 2 GPU server.

### 5. Reader training
We trained reader model for large datasets using a single 8 GPU x 32 GB server. All the default parameters are already set to our best NQ settings.
Please also download data.gold_passages_info.nq_train & data.gold_passages_info.nq_dev resources for NQ datatset - they are used for special NQ only heuristics when preprocessing the data for the NQ reader training. If you already run reader trianign on NQ data without gold_passages_src & gold_passages_src_dev specified, please delete the corresponding .pkl files so that thye will be re-generated.

```bash
python train_extractive_reader.py \
	encoder.sequence_length=350 \
	train_files={path to the retriever train set results file} \
	dev_files={path to the retriever dev set results file}  \
	gold_passages_src={path to data.gold_passages_info.nq_train file} \
	gold_passages_src_dev={path to data.gold_passages_info.nq_dev file} \
	output_dir={path to output dir}
```

We found that using the learning rate above works best with static schedule, so one needs to stop training manually based on evaluation performance dynamics.
Our best results were achieved on 16-18 training epochs or after ~60k model updates.

We provide all input and intermediate results for e2e pipeline for NQ dataset and most of the similar resources for Trivia.

## Misc.
- TREC validation requires regexp based matching. We support only retriever validation in the regexp mode. See --match parameter option.
- WebQ validation requires entity normalization, which is not included as of now.

## License
DPR is CC-BY-NC 4.0 licensed as of now.
