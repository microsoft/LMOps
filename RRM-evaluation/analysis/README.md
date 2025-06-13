# Visualizations & Eval Analysis for HERM

We're going to add visualizations for both the eval. data and results here.
So far, we have the following tools:

### Convert BoN outputs to AlpacaEval format
```
python analysis/bon_to_alpacaeval.py --generation_model=zephyr-7b --reward_model=OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5
```

### Get per task distribution
```
python analysis/plot_per_subset_dist.py --output_dir=plots/whisker
```

### Get benchmark results
This prints out the RewardBench results in a Markdown or LaTeX table. Note that you need to pass an API token to the `HF_COLLAB_TOKEN` environment variable.
```
# Use --render_latex for LaTeX output
python analysis/get_benchmark_results.py
```

Below is a snippet of the output for the RewardBench - General results:

| model                                            |   average |   alpacaeval |   mt-bench |   llmbar |   refusals |    hep |
|--------------------------------------------------|-----------|--------------|------------|----------|------------|--------|
| berkeley-nest/Starling-RM-7B-alpha               |      0.74 |         0.89 |       0.84 |     0.45 |       0.7  |   0.8  |
| openbmb/UltraRM-13b                              |      0.68 |         0.98 |       0.93 |     0.54 |       0.08 |   0.86 |
| stabilityai/stablelm-zephyr-3b                   |      0.64 |         0.9  |       0.84 |     0.52 |       0.3  |     |
| OpenAssistant/reward-model-deberta-v3-large-v2   |      0.64 |         0.88 |       0.81 |     0.25 |       0.61 |   0.65 |
| OpenAssistant/oasst-rm-2-pythia-6.9b-epoch-1     |      0.63 |         0.95 |       0.78 |     0.36 |       0.42 |     |
| OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 |      0.62 |         0.86 |       0.79 |     0.5  |       0.35 |     |
| llm-blender/PairRM-hf                            |      0.6  |         0.85 |       0.86 |     0.53 |       0.13 |   0.64 |
| weqweasdas/hh_rlhf_rm_open_llama_3b              |      0.54 |         0.79 |       0.72 |     0.41 |       0.22 |     |
| stanfordnlp/SteamSHP-flan-t5-xl                  |      0.48 |         0.85 |       0.7  |     0.38 |       0.01 |   0.48 |

Also, these can be visualized as a distribution with
```
python analysis/plot_per_subset_dist.py
```


### Per token uterrance reward
This returns the reward per-token to show how the reward evolves over a piece of text.
Use of this tool requires installation of `spacy-alignments`:
```
pip install spacy-alignments
```
Then,
```
python analysis/per_token_reward.py --model=OpenAssistant/reward-model-deberta-v3-large-v2  --text="I love to walk the dog, what do you like?"
```
E.g. with OpenAssistant/reward-model-deberta-v3-large-v2
```
Reward: -0.544 | Substring: I
Reward: -0.556 | Substring: I love
Reward: -0.566 | Substring: I love to
Reward: 0.099 | Substring: I love to walk
Reward: 0.096 | Substring: I love to walk the
Reward: 0.092 | Substring: I love to walk the dog
Reward: 0.09 | Substring: I love to walk the dog,
Reward: 0.087 | Substring: I love to walk the dog, what
Reward: 0.085 | Substring: I love to walk the dog, what do
Reward: 0.089 | Substring: I love to walk the dog, what do you
Reward: 0.09 | Substring: I love to walk the dog, what do you like
Reward: 0.093 | Substring: I love to walk the dog, what do you like?
```
### Model usage within eval. dataset
To run this, execute:
```
python analysis/draw_model_histogram output.png --log_scale
```
![output](https://github.com/allenai/herm/assets/10695622/e5aa4c0f-83de-4997-8307-f49c22456671)

This will also return the following table by default:

| Model | Total | chosen_model | rejected_model |
| --- | --- | --- | --- |
| human | 2107 | 985 | 1122 |
| unknown | 838 | 419 | 419 |
| GPT-4 | 516 | 466 | 50 |
| Llama-2-70b-chat | 251 | 163 | 88 |
| Mistral-7B-Instruct-v0.1 | 244 | 117 | 127 |
| dolphin-2.0-mistral-7b | 208 | 0 | 208 |
| GPT4-Turbo | 100 | 100 | 0 |
| alpaca-7b | 100 | 0 | 100 |
| tulu-2-dpo-70b | 95 | 95 | 0 |
| davinci-003 | 95 | 0 | 95 |
| guanaco-13b | 95 | 0 | 95 |
| zephyr-7b-beta | 87 | 69 | 18 |
| ChatGLM2 | 52 | 0 | 52 |
| vicuna-7b | 38 | 0 | 38 |
| GPT-3.5-Turbo | 29 | 22 | 7 |
| claude-v1 | 23 | 9 | 14 |
| dolly-v2-12b | 19 | 1 | 18 |
| fastchat-t5-3b | 18 | 2 | 16 |
| llama-13b | 17 | 1 | 16 |
| falcon-40b-instruct | 11 | 8 | 3 |
| rwkv-4-raven-14b | 11 | 1 | 10 |
| stablelm-tuned-alpha-7b | 11 | 0 | 11 |
| alpaca-13b | 10 | 3 | 7 |
| chatglm-6b | 8 | 4 | 4 |
| mpt-30b-instruct | 7 | 6 | 1 |
| h2ogpt-oasst-open-llama-13b | 6 | 4 | 2 |
| palm-2-chat-bison-001 | 6 | 4 | 2 |
| gpt4all-13b-snoozy | 6 | 5 | 1 |
| guanaco-65b | 5 | 5 | 0 |
| oasst-sft-4-pythia-12b | 5 | 2 | 3 |
| Llama-2-7b-chat | 5 | 3 | 2 |
| mpt-30b-chat | 5 | 5 | 0 |
| mpt-7b-chat | 5 | 3 | 2 |
| guanaco-33b | 4 | 4 | 0 |
| Llama-2-13b-chat | 4 | 3 | 1 |
| vicuna-13b-v1.3 | 4 | 3 | 1 |
| koala-13b | 4 | 4 | 0 |
| baize-v2-13b | 4 | 4 | 0 |
| oasst-sft-7-llama-30b | 4 | 4 | 0 |
| nous-hermes-13b | 4 | 2 | 2 |
| vicuna-7b-v1.3 | 3 | 2 | 1 |
| claude-instant-v1 | 3 | 3 | 0 |
| wizardlm-30b | 3 | 3 | 0 |
| wizardlm-13b | 3 | 1 | 2 |
| tulu-30b | 2 | 2 | 0 |
| vicuna-33b-v1.3 | 1 | 1 | 0 |

Total number of models involved: 44
