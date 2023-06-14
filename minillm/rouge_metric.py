import string
import json
import os
import argparse
from rouge_score import rouge_scorer
from transformers import AutoTokenizer


default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def rouge(prediction, ground_truth, xlingual=False):
    scorer = default_rouge_scorer
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    # assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    
    min_length = min(len((predictions)), len(references))
    predictions = predictions[:min_length]
    references = references[:min_length]
    
    em, rougeL = 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        em += metric_max_over_ground_truths(
            exact_match, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rouge, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    em = 100.0 * em / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": em, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_file", required=True,
        help="Jsonl file with each line corresponding to a prediction. " 
             "Each json object should have an `id` and a `prediction` key.")
    parser.add_argument(
        "--reference_file", required=True,
        help="Jsonl file with each line corresponding to a reference. " 
             "Each json object should have an `id` and a `references` key. "
             "`task_id`, `task_category` and `task_track` are optional, which will be used to "
             "compute the per-task performance, per-category performance and the performance for default (english) / xlingual Tracks.")
    parser.add_argument(
        "--output_file",
        help="Jsonl file to write the results to.")
    parser.add_argument(
        "--model_name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    references = []
    with open(args.reference_file) as fin:
        for line in fin:
            instance = json.loads(line)
            if isinstance(instance["output"], list):
                references.append(instance["output"])
            else:
                references.append([instance["output"]])

    predictions = []
    with open(args.prediction_file) as fin:
        for line in fin:
            prediction = json.loads(line)
            predictions.append(prediction["text"])

    predictions = predictions[:1000]

    references = references[:len(predictions)]

    results = compute_metrics(predictions, references, xlingual=False)

    print(results)

    if args.output_file:
        os.makedirs(args.output_file, exist_ok=True)
        with open(os.path.join(args.output_file, f"{args.model_name}.json"), "w") as fout:
            json.dump(results, fout, indent=2)
            