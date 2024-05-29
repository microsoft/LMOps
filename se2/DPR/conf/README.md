## Hydra

[Hydra](https://github.com/facebookresearch/hydra) is an open-source Python
framework that simplifies the development of research and other complex
applications. The key feature is the ability to dynamically create a
hierarchical configuration by composition and override it through config files
and the command line. 

## DPR configuration
All DPR tools configuration parameters are now split between different config groups and you can either modify them in the config files or override from command line.

Each tools's (train_dense_encoder.py, generate_dense_embeddings.py, dense_retriever.py and train_reader.py) main method has now a hydra @hydra.main decorator with the name of the configuration file in the conf/ dir.
For example, dense_retriever.py takes all its parameters from conf/dense_retriever.yaml file.
Every tool's configuration files refers to other configuration files via "defaults:" parameter. 
It is called a [configuration group](https://hydra.cc/docs/tutorials/structured_config/config_groups) in Hydra.

Let's take a look at dense_retriever.py's configuration:


```yaml

defaults:
  - encoder: hf_bert
  - datasets: retriever_default
  - ctx_sources: default_sources

indexers:
  flat:
    _target_: dpr.indexer.faiss_indexers.DenseFlatIndexer

  hnsw:
    _target_: dpr.indexer.faiss_indexers.DenseHNSWFlatIndexer

  hnsw_sq:
    _target_: dpr.indexer.faiss_indexers.DenseHNSWSQIndexer

...
qa_dataset:
...
ctx_datatsets:
...
indexer: flat
...

```

"  - encoder: " - a configuration group that contains all parameters to instantiate the encoder. The actual parameters are located in conf/encoder/hf_bert.yaml file.
If you want to override some of them, you can either 
- Modify that config file
- Create a new config group file under  conf/encoder/ folder and enable to use it by providing encoder={your file name} command line argument
- Override specific parameter from command line. For example: encoder.sequence_length=300

"  - datasets:" - a configuration group that contains a list of all possible sources of queries for evaluation. One can find them in conf/datasets/retriever_default.yaml file.
One should specify the dataset to use by providing qa_dataset parameter in order to use one of them during evaluation. For example, if you want to run the retriever on NQ test set, set qa_dataset=nq_test as a command line parameter.

It is much easier now to use custom datasets, without the need to convert them to DPR format. Just define your own class that provides relevant __getitem__(), __len__() and load_data() methods (inherit from QASrc).

"  - ctx_sources: " - a configuration group that contains a list of all possible passage sources.  One can find them in conf/ctx_sources/default_sources.yaml file.
One should specify a list of names of the passages datasets as ctx_datatsets parameter. For example, if you want to use dpr's old wikipedia passages, set ctx_datatsets=[dpr_wiki]. 
Please note that this parameter is a list and you can effectively concatenate different passage source into one. In order to use multiple sources at once, one also needs to provide relevant embeddings files in encoded_ctx_files parameter, which is also a list.


"indexers:" - a parameters map that defines various indexes. The actual index is selected by indexer parameter which is 'flat' by default but you can use loss index types by setting indexer=hnsw or indexer=hnsw_sq in the command line.

Please refer to the configuration files comments for every parameter.
