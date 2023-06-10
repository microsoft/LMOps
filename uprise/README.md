# UPRISE: Universal Prompt Retrieval for LLMs

This repository contains the code implementation and the trained prompt retriever for our paper [UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation](https://arxiv.org/abs/2303.08518), as well as explorations based on UPRISE, such as few-shot prompt retriever and chain-of-thought prompting...

We propose **UPRISE** (**U**niversal **P**rompt **R**etrieval for **I**mproving zero-**S**hot **E**valuation), which tunes a lightweight and versatile retriever that automatically retrieves prompts for a given zero-shot task input. Specifically, we demonstrate universality in a cross-task and cross-model scenario: UPRISE tunes a prompt retriever on multiple tasks with a small frozen LLM, but conducts inference on unseen task types with a different larger LLM.

<img src="https://user-images.githubusercontent.com/4668004/226875115-36dbee82-40a0-4e42-b30e-047a8d756fc2.png" width="400" height="224">

**************************** **Updates** ****************************
* 05/29: Updated training and evaluation code, to easily [train your UPRISE](#training) and [explore more of UPRISE](#exploration).
* 03/15: Released [our paper](https://arxiv.org/abs/2303.08518), code and model.

# Contents
* [Setup](#setup)
* [Evaluation](#evaluation)
* [Training](#training)
* [Exploration](#exploration)
* [Citation](#citation)

# Environment Setup <a name="setup"></a>
```bash
bash install.sh 
```

# Evaluate Our UPRISE on Any Task with Any LLM <a name="evaluation"></a>
In this section, we will introduce how to use our pre-trained retriever to improve any LLM on any task.
## 1. Download Retriever and Prompt Pool
Download the [retriever](https://drive.google.com/file/d/1fd6wYiP8LMRfM1f2oiFSn4t5jb_xLuo-/view?usp=share_link) tuned on all the tasks listed in Appendix A of [our paper](https://arxiv.org/abs/2303.08518), and the pre-constructed [prompt pool](https://drive.google.com/file/d/1NT3dYvheoFGnP3wTlGTJ9dyXj0-3hvTe/view?usp=share_link).

After downloading the retriever and prompt pool, encode the prompt pool with the prompt encoder using the following command:
```bash
export RETRIEVER=[DOWNLOADED_CKPT_PATH] # path to the downloaded retriever checkpoint
export PROMPT_POOL=[DOWNLOADED_POOL_PATH] # path to the downloaded prompt pool
export CACHE_DIR=[CAHCHE_DIR] # directory path for caching the LLM checkpoints, task datasets, etc.

python DPR/generate_dense_embeddings.py \
	 model_file=${RETRIEVER} \
	 ctx_src=dpr_uprise shard_id=0 num_shards=1 \
	 out_file=$PWD/my_data/experiment/uprise/dpr_enc_index \
	 ctx_sources.dpr_uprise.prompt_pool_path=${PROMPT_POOL} \
	 ctx_sources.dpr_uprise.prompt_setup_type=qa \
	 encoder.cache_dir=${CACHE_DIR} \
	 hydra.run.dir=$PWD/my_data/experiment/uprise
```
The encoded pool and the experiment logs are in `my_data/experiment/uprise`.
Once the prompt pool is encoded, you can directly use it to test on many tasks with many LLMs.

## 2. [OPTIONAL] Define Your Testing Task
UPRISE supports any task, and we have defined all the tasks used in our paper in [tasks.py](./DPR/dpr/utils/tasks.py), you can also simply add any other tasks for testing.

We divide all the tasks into two question types: multiple choice and text completion, and use different methods to implement them.

**Multiple Choice** is the question to choose one correct completion from several options. You can follow [SNLI](./DPR/dpr/utils/tasks.py#L322-L390) to implement your multiple choice task.

**Text Completion** is the question to do free-form completion. You can follow [SQuADv1](./DPR/dpr/utils/tasks.py#L556-L619) to implement your text completion task.

## 3. [OPTIONAL] Define Your Evaluation Metric
There are several metrics already defined in [metric.py](./src/utils/metric.py). However, if you want to add a new metric, you can follow the steps below:

1. Define your metric function. For example, if you want to add the SQuAD metric, you can define the function as follows:
```python
def squad(labels, preds):
    """Computes SQuAD metrics, maximizing over answers per question.
    Args:
    labels: list of lists of strings
    preds: list of strings
    Returns:
    dict with score_key: squad score across all labels and predictions
    """
    labels = [[qa_utils.normalize_squad(t) for t in u] for u in labels]
    preds = [qa_utils.normalize_squad(p) for p in preds]
    em,f1=qa_utils.qa_metrics(labels, preds) #em,f1
    return em,f1
```

2. Add your metric function to the `compute_metrics` function. For instance, you can add the following code to the [compute_metrics](./src/utils/metric.py#L140-L167) function:
```python
if metric=='squad': # the metric name should be the same with the set metric in your task class in task.py
    em,f1=squad(labels=labels, preds=preds)
    return {'em':em, 'f1':f1}
```
Note that you should name your metric function with a unique name and also use the same name in the compute_metrics function to call your metric.

## 4. Run inference with LLMs
UPRISE supports any LLM, such as ChatGPT, GPT-3, LLaMA, OPT, BLOOM and GPT-Neo, and you can test UPRISE on other LLMs with HuggingFace🤗 or OpenAI API.

**LLMs available in HuggingFace🤗**

To run inference with a LLM available in HuggingFace, use the following script:
```bash
export TASK=rte # task name for evaluation, should be the same as the name in the task.py file
export LLM="EleutherAI/gpt-neo-2.7B" # LLM for inference

bash inference_hf.sh
```

**LLMs with OpenAI API**

Before running inference with OpenAI LLMs, run the above script with a Huggingface LLM, as we need to reuse the model prediction file.

Note that our inference code has been verified on the LLMs belonging to the GPT-3 series only. For other models such as ChatGPT, please refer to the [official documentation](https://platform.openai.com/docs/api-reference/introduction).

Please first setup the environment in [this repo](https://github.com/microsoft/semantic_parsing_with_constrained_lm). And then run:
```bash
export OPENAI_TOKEN=[OPENAI_KEY] # your OpenAI API key
export ENGINE='ada' # LLM engine name, such as davinci, ada, curie, babbage

bash inference_openai.sh
```

## FAQ: Why do the task scores of UPRISE appear significantly higher when testing tasks in the [tasks.py](./DPR/dpr/utils/tasks.py) compared to the scores reported in the [paper](https://arxiv.org/abs/2303.08518)?

The disparity in task scores arises from the difference in training methodologies. The downloaded retriever included in the repository is trained on all the tasks specified in [tasks.py](./DPR/dpr/utils/tasks.py). Consequently, when you utilize this retriever to test tasks in the same file, it does not adhere to the cross-task setting outlined in our paper. As a result, the observed task scores are notably higher than those reported in the paper.

# Train and Evaluate Your UPRISE <a name="training"></a>
In the following, we will detail on how to train and evaluate your universal prompt retriever.

## 1. Define Your Task Clusters for Training and Evaluation
As mentioned before, we have defined many tasks in [tasks.py](./DPR/dpr/utils/tasks.py), you can also follow `step 2` and `step 3` in the [Evaluation](#evaluation) section to implement your tasks.
And then add your tasks in the [train_cluster_map](./DPR/dpr/utils/tasks.py#L11-L28) and [test_cluster_map](./DPR/dpr/utils/tasks.py#L29-L46) like this:
```python
train_cluster_map = {
    "train_example_1": ["rte"],
    "train_example_2": ["copa", "piqa"],
}
test_cluster_map = {
    "test_example_1": ["arc_e"],
    "test_example_2": ["mrpc"],
}
```

## 2. Get Train and Inference Commands
It's very easy to customize your own UPRISE with [get_cmds.py](get_cmds.py) which provides a detailed list of every single argument for the retriever. It also includes information on how to use BM25 and (S)BERT to retrieve from the prompt pool and run ablations. Let's check them out!

An example command is:
```bash
TRAIN_CLUSTERS='train_example_1+train_example_2' # use `+` to concatenate your train clusters as a string
TEST_CLUSTERS='test_example_1+test_example_2'  # use `+` to concatenate your test clusters as a string
SCORE_LLM='EleutherAI/gpt-neo-2.7B'  # LLM to score the data
INF_LLM='decapoda-research/llama-7b-hf' # LLM for inference
OUTPUT_DIR=my_data
python get_cmds.py \
    --output_dir ${OUTPUT_DIR} \
    --train_clusters ${TRAIN_CLUSTERS} \
    --test_clusters ${TEST_CLUSTERS} \
    --scr_model ${SCORE_LLM} \
    --inf_model ${INF_LLM} \
    --multi_task
```

Then you can get all the train and inference cmds under the `${OUTPUT_DIR}/experiment/train_${TRAIN_CLUSTERS}_test${TEST_CLUSTERS}/` folder:
```bash
OUTPUT_FOLDER=${OUTPUT_DIR}/experiment/train_${TRAIN_CLUSTERS}_test${TEST_CLUSTERS}
bash ${OUTPUT_FOLDER}/train.sh # score data and train a retriever
bash ${OUTPUT_FOLDER}/inference.sh # use the trained retriever for inference
```

# Explore More of UPRISE: Few-shot ICL, CoT Prompts, Other Retrievers <a name="exploration"></a>
## Few-shot Demonstration Retrieval for In-context Learning (ICL)
In our paper, we conduct cross-task validation to demonstrate zero-shot transferrability of UPRISE. However, you can also use UPRISE to train a task-specific retriever to retrieve few-shot demonstrations for in-context learning. To do this, set the task(s) for training to be the same as the task(s) for testing. By doing so, the trained retriever can select demonstrations from the task's training set.
```python
train_cluster_map = {
    "train_cluster": ["your_task"], 
}
test_cluster_map = {
    "test_cluster": ["your_task"], 
}
```
Similary, get all the cmds with the [get_cmds.py](get_cmds.py), remember NOT to set the `multi_task` option when there is only one task for training, and that's all!
```bash
TRAIN_CLUSTERS="train_cluster"
TEST_CLUSTERS="test_cluster" 
SCORE_LLM='EleutherAI/gpt-neo-2.7B' 
INF_LLM='decapoda-research/llama-7b-hf'
python get_cmds.py \
    --train_clusters ${TRAIN_CLUSTERS} \
    --test_clusters ${TEST_CLUSTERS} \
    --scr_model ${SCORE_LLM} \
    --inf_model ${INF_LLM}
```

```bash
OUTPUT_FOLDER=${OUTPUT_DIR}/experiment/train_${TRAIN_CLUSTERS}_test${TEST_CLUSTERS}
bash ${OUTPUT_FOLDER}/train.sh
bash ${OUTPUT_FOLDER}/inference.sh
```
## Incorporate Chain-of-Thought (CoT) Demonstrations in UPRISE
In this paper, we include standard in-context learning demonstrations in our prompt pool as one use case, however, we can easily incorporate chain-of-thought prompts in UPRISE with some minor modifications.

Here's an example of how to implement [PubMedQA](https://arxiv.org/pdf/1909.06146.pdf) as a CoT task:
1. Implement your CoT task in [tasks.py](./DPR/dpr/utils/tasks.py) using the [pubmed_qa](./DPR/dpr/utils/tasks.py#L1747-L1825) task implementation as a reference.
2. Add a metric for evaluating the CoT task in [metric.py](./src/utils/metric.py) following the example of [pubmed_qa_acc](./src/utils/metric.py#L95-L119).
3. Then you can get the training and inference commands as in the [Training](#training) section.
- **✨ Tips**: For task datasets that do not include chain-of-thought explanations, you can utilize the LLM itself to generate chain-of-thought demonstrations and include them in the prompt pool.

## Use Other Sentence Embedding Models to Initilize the Retriever
In our pretrained UPRISE, we used BERT-base to initialize the retriever. However, you can use other sentence embedding models such as [E5](https://github.com/microsoft/unilm/tree/master/e5) and [INSTRUCTOR](https://github.com/HKUNLP/instructor-embedding) to initialize the retriever to see if they perform better for your specific task or use case. To do so, you can add your desired models to `DPR/dpr/models` and modify the configs under `DPR/conf` accordingly. 

# Bugs or Questions?
If you have any question related to the code or the paper, feel free to open an issue or email Daixuan (`daixuancheng6@gmail.com`). Please try to specify the problem with details so we can help you better and quicker.

# License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

# Citation <a name="citation"></a>
If you find our work helpful, please cite us:
```bibtex
@inproceedings{UPRISE,
  title={UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation},
  author={Daixuan Cheng and Shaohan Huang and Junyu Bi and Yuefeng Zhan and Jianfeng Liu and Yujing Wang and Hao Sun and Furu Wei and Denvy Deng and Qi Zhang},
  url={https://arxiv.org/abs/2303.08518},
  year={2023},
}
```
