# Break Evaluator
Evaluator for the [Break](https://github.com/allenai/Break) dataset (AI2 Israel).  
Used in both the Break and Break High-level leaderboards.

## Example
```bash
% PYTHONPATH="." python3.7 scripts/evaluate_predictions.py 
--dataset_file=/labels/labels.csv \
--preds_file=/predictions/predictions.csv \
--no_cache \
--output_file_base=/results/results \
--metrics ged_scores exact_match sari normalized_exact_match \
				
% cat results/results_metrics.json
{"exact_match": 0.24242424242424243, "sari": 0.7061778423719823, "ged": 0.4089606835211786, "normalized_exact_match": 0.32323232323232326}
```

## Usage

### Input
The evaluation script recieves as input a Break `dataset_file` which is a CSV file containing the correct *labels*. Additionally, it should receive `preds_file`, a CSV file containing a model's *predictions*, ordered according to `dataset_file`. The `output_file_base` indicates the file to which the evaluation output be saved. Last `metrics` indicates which evaluation metrics should be included out of `ged_scores, exact_match, sari, normalized_exact_match`.

The `tmp` directory contains examples of [`dataset_file`](https://github.com/allenai/break-evaluator/blob/master/tmp/labels/labels.csv) and [`preds_file`](https://github.com/allenai/break-evaluator/blob/master/tmp/predictions/predictions.csv).

### Output
The evaluation output will be saved to `output_file_base_metrics.json`


## Setup
To run the evaluation script locally, using a *conda virtual environment*, do the following:

1. Create a virtual environment
```
conda create -n [ENV_NAME] python=3.7
conda activate [ENV_NAME]
```

2. Install requirements
```
pip install -r requirements.txt 
python -m spacy download en_core_web_sm
```

3. Run in shell
```
PYTHONPATH="." python3.7 scripts/evaluate_predictions.py 
--dataset_file=/labels/labels.csv \
--preds_file=/predictions/predictions.csv \
--no_cache \
--output_file_base=/results/results \
--metrics ged_scores exact_match sari normalized_exact_match \
```


## Docker
We build an evaluator image using Docker, and the specified Dockerfile.

### Build
To build the break-evaluator image:
```
docker build --tag break-evaluator .
```

### Run
Our evaluator should receive three files as input, the dataset true labels, the model's prediction file and the path to the output file. We therefore *bind mount* the relevant files when using `docker run`. 
The specific volume mounts, given our relevant files are storem in `tmp`, will be:
```
-v "$(pwd)"/tmp/results/:/results:rw
-v "$(pwd)"/tmp/predictions/:/predictions:ro
-v "$(pwd)"/tmp/labels/:/labels:ro
```

The full run command being:
```
sudo docker run -it -v "$(pwd)"/tmp/results/:/results:rw -v "$(pwd)"/tmp/predictions/:/predictions:ro -v "$(pwd)"/tmp/labels/:/labels:ro break-evaluator bash -c "python3.7 scripts/evaluate_predictions.py --dataset_file=/labels/labels.csv --preds_file=/predictions/predictions.csv --no_cache --output_file_base=/results/results --metrics ged_scores exact_match sari normalized_exact_match"
```


## Beaker
To add a Beaker image of the evaluator run:
```
beaker image create -n break-evaluator-YYYY-MM-DD break-evaluator:latest
```



## Evaluation Metircs
To learn more about the evaluation metrics used for [Break](https://allenai.github.io/Break/), please refer to the paper ["Break It Down: A Question Understanding Benchmark" (Wolfson et al., TACL 2020)](https://arxiv.org/abs/2001.11770).  
The *"Normalized Exact Match"* metric, is a newly introduced evaluation metric for QDMR that will be included in future work. It compares two QDMRs by normalizing their respective graphs: further decomposing steps; ordering chains of "filter" operations; lemmatizing step noun phrases; etc. 