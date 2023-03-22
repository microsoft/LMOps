from typing import Dict, Tuple
import numbers
from itertools import zip_longest

import argparse
import os
import random
import re
import numpy as np
import pandas as pd
import json


from evaluation.decomposition import Decomposition
from evaluation.graph_matcher import GraphMatchScorer, get_ged_plus_scores
from evaluation.sari_hook import get_sari
from evaluation.sequence_matcher import SequenceMatchScorer
from evaluation.normal_form.normalized_graph_matcher import NormalizedGraphMatchScorer
import evaluation.normal_form.normalization_rules as norm_rules


pd.set_option('display.max_colwidth', -1)


def evaluate(ids, questions, decompositions, golds, metadata,
             output_path_base,
             metrics=None):
    decompositions_str = [d.to_string() for d in decompositions]
    golds_str = [g.to_string() for g in golds]

    # calculating exact match scores
    exact_match = get_exact_match(decompositions_str, golds_str) \
        if (metrics is None) or 'exact_match' in metrics else None

    # evaluate using SARI
    sari = get_sari_score(decompositions_str, golds_str, questions) \
        if (metrics is None) or 'sari' in metrics else None

    # evaluate using sequence matcher
    match_ratio = get_match_ratio(decompositions_str, golds_str) \
        if (metrics is None) or 'match' in metrics else None
    structural_match_ratio = get_structural_match_ratio(decompositions_str, golds_str) \
        if (metrics is None) or 'structural_match' in metrics else None

    # evaluate using graph distances
    graph_scorer = GraphMatchScorer()
    decomposition_graphs = [d.to_graph() for d in decompositions]
    gold_graphs = [g.to_graph() for g in golds]

    # ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs)

    # structural_ged_scores = graph_scorer.get_edit_distance_match_scores(decomposition_graphs, gold_graphs,
    #                                                                     structure_only=True)
    # ged_plus_scores = get_ged_plus_scores(decomposition_graphs, gold_graphs,
    #                                       exclude_thr=5, num_processes=num_processes)

    # calculate normalized match scores
    normalize_scorer = NormalizedGraphMatchScorer()

    def try_invoke(func, graph, default=None):
        try:
            return func(graph)
        except Exception as ex:
            return default

    decomposition_norm_graphs = [try_invoke(normalize_scorer.normalize_graph, g, default=g) for g in
                                 decomposition_graphs]
    decomposition_norm_str = [try_invoke(lambda x: Decomposition.from_graph(x).to_string(), g) for g in
                              decomposition_norm_graphs]
    gold_norm_graphs = [try_invoke(normalize_scorer.normalize_graph, g, default=g) for g in gold_graphs]
    gold_norm_str = [try_invoke(lambda x: Decomposition.from_graph(x).to_string(), g) for g in gold_norm_graphs]

    normalized_exact_match = skip_none(get_exact_match, decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_exact_match' in metrics else None
    normalized_sari = skip_none(get_sari_score, decomposition_norm_str, gold_norm_str, questions) \
        if (metrics is None) or 'normalized_sari' in metrics else None
    normalized_match_ratio = skip_none(get_match_ratio, decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_match' in metrics else None
    normalized_structural_match_ratio = skip_none(get_structural_match_ratio, decomposition_norm_str, gold_norm_str) \
        if (metrics is None) or 'normalized_structural_match' in metrics else None

    evaluation_dict = {
        "id": ids,
        "question": questions,
        "gold": golds_str,
        "prediction": decompositions_str,
        "exact_match": exact_match,
        "match": match_ratio,
        "structural_match": structural_match_ratio,
        "sari": sari,
        # "ged": ged_scores,
        # "structural_ged": structural_ged_scores,
        # "ged_plus": ged_plus_scores,

        "normalized_exact_match": normalized_exact_match,
        "normalized_match": normalized_match_ratio,
        "normalized_structural_match": normalized_structural_match_ratio,
        "normalized_sari": normalized_sari,
    }
    evaluation_dict = {k: v for k, v in evaluation_dict.items() if v is not None}
    num_examples = len(questions)
    print_first_example_scores(evaluation_dict, min(5, num_examples))
    mean_scores = print_score_stats(evaluation_dict)

    if output_path_base:
        write_evaluation_output(output_path_base, num_examples, **evaluation_dict)
        ### Addition write the mean scores json
        write_evaluation_results(output_path_base, mean_scores)

    if metadata is not None:
        #metadata = metadata[metadata["question_text"].isin(evaluation_dict["question"])]
        metadata = metadata[metadata['question_id'].isin(evaluation_dict['id'])]
        metadata["dataset"] = metadata["question_id"].apply(lambda x: x.split("_")[0])
        metadata["num_steps"] = metadata["decomposition"].apply(lambda x: len(x.split(";")))
        score_keys = [key for key in evaluation_dict if key not in ["id", "question", "gold", "prediction"]]
        for key in score_keys:
            metadata[key] = evaluation_dict[key]

        for agg_field in ["dataset", "num_steps"]:
            df = metadata[[agg_field] + score_keys].groupby(agg_field).agg("mean")
            print(df.round(decimals=3))

    return mean_scores


def skip_none(func, *args, **kwargs):
    zipped = list(zip_longest(*args))
    none_ids = [i for i, x in enumerate(zipped) if None in x]
    args_ = tuple([x for i,x in enumerate(a) if i not in none_ids] for a in args)
    res = func(*args_, **kwargs)

    combined = []
    none_i = 0
    res_i = 0
    for i in range(len(zipped)):
        if none_i < len(none_ids) and (i == none_ids[none_i]):
            combined.append(None)
            none_i += 1
        else:
            combined.append(res[res_i])
            res_i += 1
    return combined


def get_exact_match(decompositions_str:[str], golds_str:[str]):
    return [d.lower() == g.lower() for d, g in zip(decompositions_str, golds_str)]


def get_sari_score(decompositions_str: [str], golds_str: [str], questions: [str]):
    sources = [q.split(" ") for q in questions]
    predictions = [d.split(" ") for d in decompositions_str]
    targets = [[g.split(" ")] for g in golds_str]
    sari, keep, add, deletion = get_sari(sources, predictions, targets)
    return sari


def get_match_ratio(decompositions_str: [str], golds_str: [str]):
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    return sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                            processing="base")


def get_structural_match_ratio(decompositions_str: [str], golds_str: [str]):
    sequence_scorer = SequenceMatchScorer(remove_stop_words=False)
    return sequence_scorer.get_match_scores(decompositions_str, golds_str,
                                            processing="structural")


def print_first_example_scores(evaluation_dict, num_examples):
    for i in range(num_examples):
        print("evaluating example #{}".format(i))
        for k,v in evaluation_dict.items():
            if isinstance(v[i], numbers.Number):
                print("\t{}: {}".format(k, round(v[i], 3)))
            else:
                print("\t{}: {}".format(k, v[i]))


def print_score_stats(evaluation_dict):
    skiped_samples = {}
    mean_scores = {}

    print("\noverall scores:")
    for key in evaluation_dict:
        # ignore keys that do not store scores
        if key in ["id", "question", "gold", "prediction"]:
            continue
        score_name, scores = key, evaluation_dict[key]

        # ignore examples without a score
        if None in scores:
            scores_ = [score for score in scores if score is not None]
            skiped_samples[key] = len(scores)-len(scores_)
        else:
            scores_ = scores

        mean_score, max_score, min_score = np.mean(scores_), np.max(scores_), np.min(scores_)
        print("{} score:\tmean {:.3f}\tmax {:.3f}\tmin {:.3f}".format(
            score_name, mean_score, max_score, min_score))
        mean_scores[score_name] = mean_score

    for score, skiped in skiped_samples.items():
        print(f"skipped {skiped} examples when computing {score}.")

    return mean_scores


def write_evaluation_output(output_path_base, num_examples, **kwargs):
    # write evaluation summary
    with open(output_path_base + '_summary.tsv', 'w') as fd:
        fd.write('\t'.join([key for key in sorted(kwargs.keys())]) + '\n')
        for i in range(num_examples):
            fd.write('\t'.join([str(kwargs[key][i]) for key in sorted(kwargs.keys())]) + '\n')

    # write evaluation scores per example
    df = pd.DataFrame.from_dict(kwargs, orient="columns")
    df.to_csv(output_path_base + '_full.tsv', sep='\t', index=False)


def write_evaluation_results(output_path_base, mean_scores):
    # write mean evaluation scores
    with open(output_path_base + '_metrics.json', 'w') as json_file:
        json.dump(mean_scores, json_file)


def format_qdmr(input:str):
    # replace multiple whitespaces with a single whitespace.
    input = ' '.join(input.split())

    # replace semi-colons with @@SEP@@ token, remove 'return' statements.
    parts = input.split(';')
    parts = [re.sub(r'return', '', part.strip().strip('\r')) for part in parts]

    # replacing references with special tokens, for example replacing #2 with @@2@@.
    parts = [re.sub(r'#(\d+)', '@@\g<1>@@', part) for part in parts]

    return Decomposition(parts)


def main(args):
    # load data
    try:
        metadata = pd.read_csv(args.dataset_file)
        ids = metadata["question_id"].to_list()
        questions = metadata["question_text"].to_list()
        golds = [format_qdmr(decomp) for decomp in metadata["decomposition"].to_list()]
    except Exception as ex:
        raise ValueError(f"Could not load dataset file {args.dataset_file}", ex)

    # load predictions
    try:
        preds_file = pd.read_csv(args.preds_file)
        predictions = [format_qdmr(pred) for pred in preds_file['decomposition'].to_list()]
    except Exception as ex:
        raise ValueError(f"Could not load predictions file {args.preds_file}", ex)

    assert len(golds) == len(predictions), "mismatch number of gold questions and predictions"

    if args.random_n and len(golds) > args.random_n:
        indices = random.sample(range(len(ids)), args.random_n)
        ids = [ids[i] for i in indices]
        questions = [questions[i] for i in indices]
        golds = [golds[i] for i in indices]
        predictions = [predictions[i] for i in indices]

    if not args.no_cache:
        norm_rules.load_cache(args.dataset_file.replace(".csv", "__cache"))
    res = evaluate(ids=ids,
                   questions=questions,
                   golds=golds,
                   decompositions=predictions,
                   metadata=metadata,
                   output_path_base=args.output_file_base,
                   metrics=args.metrics)
    if not args.no_cache:
        norm_rules.save_cache(args.dataset_file.replace(".csv", "__cache"))
    return res


def validate_args(args):
    # input question(s) for decomposition are provided.
    assert args.preds_file and args.dataset_file

    # input files exist.
    if args.dataset_file:
        assert os.path.exists(args.dataset_file)
    if args.preds_file:
        assert os.path.exists(args.preds_file)


def real_main():
    parser = argparse.ArgumentParser(description="evaluate QDMR predictions")
    parser.add_argument('--dataset_file', type=str, help='path to dataset file')
    parser.add_argument('--preds_file', type=str, help='path to a csv predictions file, with "prediction" column')
    parser.add_argument('--random_n', type=int, default=0,
                        help='choose n random examples from input file')

    parser.add_argument('--no_cache', action='store_true',
                        help="don't cache dependency parsing on normalized metrics")
    parser.add_argument('--output_file_base', type=str, default=None, help='path to output file')
    parser.add_argument('--metrics', nargs='+', default=['exact_match', 'sari', 'ged', 'normalized_exact_match'], help='path to output file')

    args = parser.parse_args()

    validate_args(args)
    res = main(args)

    # rename for AllenAI leader board
    map = {'exact_match': 'EM', 'normalized_exact_match': 'norm_EM', 'sari': 'SARI', 'ged': 'GED'}
    res = {map.get(k, k): v for k,v in res.items()}
    print(res)


if __name__ == '__main__':
    real_main()
