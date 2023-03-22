# UPRISE: Universal Prompt Retrieval for LLMs

This repository contains the code implementation and the trained prompt retriever for our paper [UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation](https://arxiv.org/abs/2303.08518). 

We propose **UPRISE** (**U**niversal **P**rompt **R**etrieval for **I**mproving zero-**S**hot **E**valuation), which tunes a lightweight and versatile retriever that automatically retrieves prompts for a given zero-shot task input. Specifically, we demonstrate universality in a cross-task and cross-model scenario: UPRISE tunes a prompt retriever on multiple tasks with a small frozen LLM, but conducts inference on unseen task types with a different larger LLM.

<img src="https://user-images.githubusercontent.com/4668004/226875115-36dbee82-40a0-4e42-b30e-047a8d756fc2.png" width="400" height="224">

## Evaluate UPRISE on Any Task with Any LLM
### 1. Environment setup

To set up the required environment, run the following command in the terminal:

```bash
bash install.sh 
```
### 2. Download Retriever and Prompt Pool

Download the [retriever](https://drive.google.com/file/d/1fd6wYiP8LMRfM1f2oiFSn4t5jb_xLuo-/view?usp=share_link) tuned on all the tasks listed in Appendix A of [our paper](https://arxiv.org/abs/2303.08518), and the pre-constructed [prompt pool](https://drive.google.com/file/d/1NT3dYvheoFGnP3wTlGTJ9dyXj0-3hvTe/view?usp=share_link).

After downloading the retriever and prompt pool, encode the prompt pool with the prompt encoder using the following command:

```bash
export RETRIEVER=[DOWNLOADED_CKPT_PATH] # path to the downloaded retriever checkpoint
export PROMPT_POOL=[DOWNLOADED_POOL_PATH] # path to the downloaded prompt pool
export CACHE_DIR=[CAHCHE_DIR] # directory path for caching the LLM checkpoints, task datasets, etc.

python DPR/generate_dense_embeddings.py \
	 model_file=${RETRIEVER} \
	 ctx_src=dpr_epr shard_id=0 num_shards=1 \
	 out_file=$PWD/experiments/dpr_enc_index \
	 ctx_sources.dpr_epr.prompt_pool_path=${PROMPT_POOL} \
	 ctx_sources.dpr_epr.cache_dir=${CACHE_DIR} \
	 hydra.run.dir=$PWD/experiments/

```
Once the prompt pool is encoded, you can directly use it to test on many tasks with many LLMs.

### 3. [OPTIONAL] Define Your Testing Task
UPRISE supports any task, and we have defined all the tasks used in our paper in [tasks.py](./DPR/dpr/utils/tasks.py), you can also simply add any other tasks for testing.

We divide all the tasks into two question types: multiple choice and text completion, and use different methods to implement them.

**Multiple Choice** is the question to choose one correct completion from several options. You can follow [SNLI](./DPR/dpr/utils/tasks.py#L252-L316) to implement your multiple choice task.

**Text Completion** is the question to do free-form completion. You can follow [SQuADv1](./DPR/dpr/utils/tasks.py#L472-L523) to implement your text completion task.

### 4. [OPTIONAL] Define Your Evaluation Metric
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

2. Add your metric function to the `compute_metrics` function. For instance, you can add the following code to the [compute_metrics](./src/utils/metric.py#L62-L80) function:
```python
if metric=='squad': # the metric name should be the same with the set metric in your task class in task.py
    em,f1=squad(labels=labels, preds=preds)
    return {'em':em, 'f1':f1}
```
Note that you should name your metric function with a unique name and also use the same name in the compute_metrics function to call your metric.

### 5. Run inference with LLMs
UPRISE supports any LLM, such as ChatGPT, GPT-3, OPT, BLOOM and GPT-Neo, and you can test UPRISE on other LLMs with HuggingFace🤗 or OpenAI API.

**LLMs available in HuggingFace**

To run inference with a LLM available in HuggingFace, use the following script:

```bash
export TASK=rte # task name for evaluation, should be the same as the name in the task.py file
export LLM="EleutherAI/gpt-neo-2.7B" # LLM for inference

bash inference_hf.sh
```

**LLMs with OpenAI API**

Before running inference with OpenAI LLMs, run the above script with a Huggingface LLM, as we need to reuse the model prediction file. Then use the following script for inference with OpenAI API:

Note that our inference code has been verified on the LLMs belonging to the GPT-3 series only. For other models such as ChatGPT, please refer to the [official documentation](https://platform.openai.com/docs/api-reference/introduction).

```bash
export OPENAI_TOKEN=[OPENAI_KEY] # your OpenAI API key
export ENGINE='ada' # LLM engine name, such as davinci, ada, curie, babbage

bash inference_openai.sh
```

## Bugs or Questions?
If you have any question related to the code or the paper, feel free to open an issue or email Daixuan (`daixuancheng6@gmail.com`). Please try to specify the problem with details so we can help you better and quicker.

## License
This project is licensed under the license found in the LICENSE file in the root directory of this source tree.

[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)


## Citation
If you find our work helpful, please cite us:

```bibtex
@inproceedings{UPRISE,
  title={UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation},
  author={Daixuan Cheng and Shaohan Huang and Junyu Bi and Yuefeng Zhan and Jianfeng Liu and Yujing Wang and Hao Sun and Furu Wei and Denvy Deng and Qi Zhang},
  url={https://arxiv.org/abs/2303.08518},
  year={2023},
}
```
