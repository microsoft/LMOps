# Constrained Language Models Yield Few-Shot Semantic Parsers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="https://avatars2.githubusercontent.com/u/9585815?s=200&v=4" width="18%">

This repository contains tools and instructions for reproducing the experiments in the paper
[**Constrained Language Models Yield Few-Shot Semantic Parsers** (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.608/).
If you use any source code or data included in this toolkit in your work, please cite the following paper.
```bib
@inproceedings{ConstrainedLMSemanticParser2021,
    title = "Constrained Language Models Yield Few-Shot Semantic Parsers",
    author = "Shin, Richard and Lin, Christopher H. and Thomson, Sam and Chen, Charles and Roy, Subhro and Platanios,  Emmanouil Antonios and Pauls, Adam and Klein, Dan and Eisner, Jason and Van Durme, Benjamin",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

## Initial set-up
- Install Poetry: https://python-poetry.org/docs/#installation.
- Install Python 3.7, which is the version of Python that has been used for developing this repository.
- Install `pipx` so that we can install command-line dependencies: https://pypa.github.io/pipx/.

First, check that we are not unintentionally in a virtualenv.
Run `poetry env info`; under "Virtualenv", it should show `Path:           NA`.
If it displays the path to an existing virtualenv, deactivate it, for example by running `deactivate` or `conda deactivate`.

Then run the following to set up the package:
```
cd semantic_parsing_with_constrained_lm
poetry config virtualenvs.in-project true --local
poetry env use <path to python3.7>
poetry install
poetry shell
```

Before running any of the commands below, run `poetry shell` to activate the virtualenv where all packages have been installed. You can `exit` to deactivate the virtualenv.

To run any experiments with GPT-3, you will need to obtain an API key from OpenAI at https://beta.openai.com/ and set an environment variable.
```
export OPENAI_API_KEY=<your API key>
```
The GPT-3 experiments use the ["davinci" engine](https://beta.openai.com/docs/engines/davinci) by default.
You can use a different engine by setting the `OPENAI_GPT3_ENGINE` environment variable.

**WARNING:**
If you run all of the experiments below using GPT-3, you will consume a very
large number of tokens, and under the default pricing of OpenAI, incur a
highly significant cost. If you would like to try a subset of the experiments instead:
- Add `--num-eval-examples N` as an argument to the commands below to only run the evaluation on the first N examples.
- Add `--exp-names [EXPERIMENT NAME]` where the experiment name is the portion of the path between `logs/` and `/results.json` in the result locations below, to only run one experiment (corresponds to one cell in a results table of the paper).

## Overnight
### Preliminary setup
Download and pre-process the data for Overnight:
```
PIPX_HOME=.pipx PIPX_BIN_DIR=.venv/bin pipx install --python <path to python3.7> codalab
python -m semantic_parsing_with_constrained_lm.domains.overnight.download_data
```

### Fine-tuning BART models
```
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/

for domain in "basketball" "blocks" "calendar" "housing" "publications" "recipes" "restaurants" "socialnetwork"; do
    python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
          --exp-names overnight_${domain}_utterance \
          --lr 1e-6 \
          --num-steps 20000 \
          --steps-per-save 20000 \
          --model-type BartV3 \
          --steps-per-decay 8 \
          --batch-size 32

    python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
          --exp-names overnight_${domain}_meaningRepresentation \
          --lr 1e-5 \
          --num-steps 20000 \
          --steps-per-save 20000 \
          --model-type BartV3 \
          --steps-per-decay 8 \
          --batch-size 32
done 
```


### Table 1
Run the following commands:
```
# GPT-3 Constrained Canonical
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.overnight_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split test-full

# BART
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.overnight_emnlp_camera_ready \
--log-dir logs/ \
--model Bart \
--eval-split test-full \
--exp-name-pattern 'overnight_Bart_test-full_.*_constrained_canonicalUtterance_train-200'
```

Then you can find the following results at the specified locations. 
- GPT-3 Constrained Canonical: `logs/overnight_GPT3_test-full_${DOMAIN}_constrained_canonicalUtterance_train-200/results.json`
- BART Constrained Canonical: `logs/overnight_Bart_test-full_${DOMAIN}_constrained_canonicalUtterance_train-200/results.json`
- All rows below the horizontal line: results were copied from the cited papers.

In the `results.json` files, each number in the table comes from `"denotation/top1"`.
`${DOMAIN}` can be one of the following: calendar, basketball, blocks, housing, publications, recipes, restaurants, socialnetwork.

### Table 2
Run the following commands:
```
# GPT-3 
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.overnight_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split test-subset \
--exp-name-pattern 'overnight_GPT3_test-subset_.*_(constrained|unconstrained-greedy)_.*_train-200' \
--exp-name-pattern 'overnight_GPT3_test-subset_.*_constrained_canonicalUtterance_train-20'

# BART
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.overnight_emnlp_camera_ready \
--log-dir logs/ \
--model Bart \
--eval-split test-full \
--exp-name-pattern 'overnight_Bart_test-full_.*_train-200'
```

Then you can find the following results at the specified locations:
- GPT-3 Constrained Canonical: `logs/overnight_GPT3_test-subset_${DOMAIN}_constrained_canonicalUtterance_train-200/results.json`
- GPT-3 Constrained Meaning: `logs/overnight_GPT3_test-subset_${DOMAIN}_constrained_meaningRepresentation_train-200/results.json`
- GPT-3 Unconstrained Canonical: `logs/overnight_GPT3_test-subset_${DOMAIN}_unconstrained_canonicalUtterance_train-200/results.json`
- GPT-3 Unconstrained Meaning: `logs/overnight_GPT3_test-subset_${DOMAIN}_unconstrained_meaningRepresentation_train-200/results.json`
- GPT-3 Constrained Canonical, n = 20: `logs/overnight_GPT3_test-subset_${DOMAIN}_constrained_canonicalUtterance_train-20/results.json`
- BART Constrained Canonical: `logs/overnight_Bart_test-full_${DOMAIN}_constrained_canonicalUtterance_train-200/results.json`
- BART Constrained Meaning: `logs/overnight_Bart_test-full_${DOMAIN}_constrained_meaningRepresentation_train-200/results.json`
- BART Unconstrained Canonical: `logs/overnight_Bart_test-full_${DOMAIN}_unconstrained_canonicalUtterance_train-200/results.json`
- BART Unconstrained Meaning: `logs/overnight_Bart_test-full_${DOMAIN}_unconstrained_meaningRepresentation_train-200/results.json`

### Figure 2
Run the following command:
```
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.overnight_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split test-subset \
--exp-name-pattern 'overnight_GPT3_test-subset_calendar_(constrained|unconstrained-beam)_.*_train-.*'
```

The data for the following series in the plot come from these files:
- CC (200): `logs/overnight_GPT3_test-subset_calendar_constrained_canonicalUtterance_train-200/results.json`
- CM (200): `logs/overnight_GPT3_test-subset_calendar_constrained_meaningRepresentation_train-200/results.json`
- UC (200): `logs/overnight_GPT3_test-subset_calendar_unconstrained-beam_canonicalUtterance_train-200/results.json`
- UM (200): `logs/overnight_GPT3_test-subset_calendar_unconstrained-beam_meaningRepresentation_train-200/results.json`
- CC (20): `logs/overnight_GPT3_test-subset_calendar_constrained_canonicalUtterance_train-20/results.json`

Each point in the series gets its value from the `"denotation/topN"` field, where N varies between 1 and 10.

## Break
### Preliminary setup
Install our copy of `break-evaluator` so that it is available on your path.
```
PIPX_HOME=.pipx PIPX_BIN_DIR=.venv/bin pipx install --python <path to python3.7> third_party/break-evaluator
```

### Fine-tuning BART
```
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
      --exp-names break_nested \
      --lr 1e-6 \
      --num-steps 20000 \
      --steps-per-save 20000 \
      --model-type BartV3 \
      --steps-per-decay 6 \
      --batch-size 32

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
      --exp-names break_QDMR \
      --lr 1e-5 \
      --num-steps 20000 \
      --steps-per-save 20000 \
      --model-type BartV3 \
      --steps-per-decay 2 \
      --batch-size 32
```

### Table 3
Run the following commands:
```
# GPT-3
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.qdmr_break_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split dev-subset 

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.qdmr_break_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split dev-full

# BART
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.qdmr_break_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split dev-full 
```

Then you can find the following results at the specified locations:
- Wolfson et al: https://leaderboard.allenai.org/break/submission/c4b3v1j22jqbqs7it330
- Coleman & Reneau: https://leaderboard.allenai.org/break/submission/c24mbsl7pqtiaau8vv00
- GPT-3 Constrained Canonical, n = 1000: `logs/break_GPT3_dev-subset_constrained_nested_train1000/results.json`
- GPT-3 Constrained Canonical, n = 100: `logs/break_GPT3_dev-subset_constrained_nested_train100/results.json`
- GPT-3 Constrained Canonical, n = 25: `logs/break_GPT3_dev-subset_constrained_nested_train25/results.json`
- GPT-3 Constrained Canonical, n = 200: `logs/break_GPT3_dev-subset_constrained_nested_train200/results.json`
- GPT-3 Constrained Meaning, n = 200: `logs/break_GPT3_dev-subset_constrained_QDMR_train200/results.json`
- GPT-3 Unconstrained Canonical, n = 200: `logs/break_GPT3_dev-subset_unconstrained-greedy_nested_train200/results.json`
- GPT-3 Unconstrained Meaning, n = 200: `logs/break_GPT3_dev-subset_unconstrained-greedy_QDMR_train200/results.json`
(horizontal rule)
- GPT-3 Constrained Canonical, n = 200, full dev set: `logs/break_GPT3_dev-full_constrained_nested_train200/results.json`
- BART Constrained Canonical, n = 200: `logs/break_Bart_dev-full_constrained_nested_train200/results.json`
- BART Constrained Meaning, n = 200: `logs/break_Bart_dev-full_constrained_QDMR_train200/results.json`
- BART Unconstrained Canonical, n = 200: `logs/break_Bart_dev-full_unconstrained-greedy_nested_train200/results.json`
- BART Unconstrained Meaning, n = 200: `logs/break_Bart_dev-full_unconstrained-greedy_QDMR_train200/results.json`

In the `results.json` files, each number in the table comes from `"break_metrics/nem @ 1"`.

### Figure 3
Run the following command:
```
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.qdmr_break_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split dev-subset \
--exp-name-pattern '.*constrained.*train(1000|200)'
```

The data for the following series in the plot come from the following files:
- CC (1000): `logs/break_GPT3_dev-subset_constrained_nested_train1000/results.json`
- CM (1000): `logs/break_GPT3_dev-subset_constrained_QDMR_train1000/results.json`
- CC (200): `logs/break_GPT3_dev-subset_constrained_nested_train200/results.json`
- CM (200): `logs/break_GPT3_dev-subset_constrained_QDMR_train200/results.json`

Each point in the series gets its value from the `"break_metrics/nem @ 1"` field, where N varies between 1 and 10.

## SMCalFlow
### Preliminary setup
Create the SCFG and preprocess the data by running the following:
```
python -m semantic_parsing_with_constrained_lm.domains.calflow.write_data
```
This script will output `semantic_parsing_with_constrained_lm/domains/calflow/grammar/grammar.scfg`
based on the .csv files in `semantic_parsing_with_constrained_lm/domains/calflow/data`.
It will also download a version of SMCalFlow pre-processed to collapse certain nested function calls
and remove re-entrancies (references to earlier nodes in the graph), and process them to create
`semantic_parsing_with_constrained_lm/domains/calflow/data/{test_200_uniform,train_300_stratified,train_1000_stratified,dev_all}.jsonl`.

### Fine-tuning BART
```
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
      --exp-names calflow_canonicalUtterance \
      --lr 1e-5 \
      --num-steps 20000 \
      --steps-per-save 20000 \
      --model-type BartV3 \
      --steps-per-decay 2 \
      --batch-size 32

python -m semantic_parsing_with_constrained_lm.finetune.lm_finetune \
      --exp-names calflow_lispress \
      --lr 1e-5 \
      --num-steps 20000 \
      --steps-per-save 20000 \
      --model-type BartV3 \
      --steps-per-decay 2 \
      --batch-size 32
```

### Table 4
Run the following commands:
```
# GPT-3
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.calflow_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split dev-full

python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.calflow_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split dev-subset

# BART
export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.calflow_emnlp_camera_ready \
--log-dir logs/ \
--model Bart \
--eval-split dev-full 
```

Then you can find the following results at the specified locations:
- GPT-3 Constrained Canonical: `logs/calflow_GPT3_dev-subset_constrained_canonicalUtterance_prompt20/results.json`
- GPT-3 Constrained Meaning: `logs/calflow_GPT3_dev-subset_constrained_lispress_prompt20/results.json`
- GPT-3 Unconstrained Canonical: `logs/calflow_GPT3_dev-subset_unconstrained-greedy_canonicalUtterance_prompt20/results.json`
- GPT-3 Unconstrained Meaning: `logs/calflow_GPT3_dev-subset_unconstrained-greedy_lispress_prompt20/results.json`
(horizontal rule)
- GPT-3 Constrained Canonical, full dev set: `logs/calflow_GPT3_dev-full_constrained_canonicalUtterance_prompt20/results.json`
- BART Constrained Canonical: `logs/calflow_Bart_dev-full_constrained_canonicalUtterance_prompt0/results.json`
- BART Constrained Meaning: `logs/calflow_Bart_dev-full_constrained_lispress_prompt0/results.json`
- BART Unconstrained Canonical: `logs/calflow_Bart_dev-full_unconstrained-greedy_canonicalUtterance_prompt0/results.json`
- BART Unconstrained Meaning: `logs/calflow_Bart_dev-full_unconstrained-greedy_lispress_prompt0/results.json`

In the `results.json` files, each number in the table comes from `"roundtrip/top1"`.

### Figure 4
Run the following commands:
```
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.calflow_emnlp_camera_ready \
--log-dir logs/ \
--model GPT3 \
--eval-split dev-full

export PRETRAINED_MODEL_DIR=facebook/bart-large
export TRAINED_MODEL_DIR=trained_models/
python -m semantic_parsing_with_constrained_lm.run_exp \
--config-name semantic_parsing_with_constrained_lm.configs.calflow_emnlp_camera_ready \
--log-dir logs/ \
--model Bart \
--eval-split dev-full  \
--exp-name-pattern '.*constrained.*'
```

The data for the following series in the plot come from the following files:
- GPT-3 CC: `logs/calflow_GPT3_dev-subset_constrained_canonicalUtterance_prompt20/results.json`
- BART CC: `logs/calflow_Bart_dev-full_constrained_canonicalUtterance_prompt0/results.json`
- BART CM: `logs/calflow_Bart_dev-full_constrained_lispress_prompt0/results.json`

Each point in the series gets its value from the `"roundtrip/topN"` field, where N varies between 1 and 10.
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
