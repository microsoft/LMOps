import os
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np
import Levenshtein

from data_utils import ChunkedDatasetBuilder, DistributedMMapIndexedDataset, best_fitting_dtype
from utils import get_tokenizer
from arguments import add_runtime_args, add_data_args, add_model_args


def add_additional_args(parser: argparse.ArgumentParser):
    parser.add_argument("--data-scorer-tokenizer-path", type=str, default=None)
    parser.add_argument("--data-scorer-model-type", type=str, default=None)
    parser.add_argument("--ds-score-path", type=str, default=None)
    parser.add_argument("--ds-ratio", type=float, default=None)
    parser.add_argument("--ds-gumbel-temperature", type=float, default=None)
    return parser


def get_args():
    parser = argparse.ArgumentParser()
    parser = add_additional_args(parser)
    parser = add_runtime_args(parser)
    parser = add_data_args(parser)
    parser = add_model_args(parser)
    args = parser.parse_args()
    return args


def add_gumbel_noise(scores, T):
    scores = scores / T
    print("Scores before noise:", scores[:10])
    u = np.random.uniform(size=np.shape(scores))
    z = -np.log(-np.log(u))
    z = torch.tensor(z, dtype=torch.float32)
    scores = scores + z
    print("Noise:", z[:10])
    print("Scores after noise:", scores[:10])    

    return scores


def sanity_check(args, tokenizer, tokenizer_cls, data):
    print("#### Sanity check ####")
    for _, _, files in os.walk(args.ds_score_path):
        for filename in files:
            if "check_indices" in filename:
                print(f"Checking {filename}")
                check_indices = torch.load(os.path.join(args.ds_score_path, filename), map_location="cpu")
                check_insts = torch.load(os.path.join(args.ds_score_path, filename.replace("indices", "insts")), map_location="cpu")
                for idx, tokens in zip(tqdm(check_indices), check_insts):
                    s = tokenizer.decode(data[idx].astype(int))[:100]
                    s_cls = tokenizer_cls.decode(tokens)[:100]
                    r = Levenshtein.ratio(s, s_cls)
                    if r < 0.9:
                        print(s)
                        print("\n\n\n")
                        print(s_cls)
                    assert r > 0.9, "The documents from different tokenizer should be the same"
    print("#### Sanity check Pass ####")
 


def main():
    args = get_args()
    output_dir = os.path.join(args.save, f"{args.data_name}-t{args.ds_gumbel_temperature}-r{args.ds_ratio}")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args)
    tokenizer_cls = get_tokenizer(
        args, model_path=args.data_scorer_tokenizer_path, model_type=args.data_scorer_model_type)

    dtype = best_fitting_dtype(tokenizer.vocab_size)

    with open(os.path.join(args.ds_score_path, "state.json")) as f:
        state = json.load(f)

    scores = []
    for sidx in tqdm(range(state["idx"]), desc="Loading Scores"):
        _scores = torch.load(os.path.join(args.ds_score_path, f"scores_{sidx}.pt"), map_location='cpu')
        scores.append(_scores)
    scores = torch.cat(scores, dim=0)

    scores = add_gumbel_noise(scores, args.ds_gumbel_temperature)

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    selected_gamma = sorted_scores[:int(args.ds_ratio * len(sorted_scores))]
    selected_indices = sorted_indices[:int(args.ds_ratio * len(sorted_scores))]

    sorted_selected_indices = torch.sort(selected_indices, descending=True).values
    sorted_selected_indices = sorted_selected_indices.tolist() # reverse order

    data = DistributedMMapIndexedDataset(args.data_dir, "data", do_probe=True)

    sanity_check(args, tokenizer, tokenizer_cls, data)

    builder = ChunkedDatasetBuilder(args.base_path, output_dir, dtype, do_shuffle=False)

    selected_num = 0
    tot = min(len(data), len(selected_gamma))

    print("Selected Indices Num:", len(sorted_selected_indices))

    idx = sorted_selected_indices.pop()
    pbar = tqdm(total=tot, desc="Train")
    for i, d in enumerate(data):
        d = d.astype(int)
        if i == idx:
            if selected_num == 0:
                print("#### Example Instance ####")
                print(d)
                print(tokenizer.decode(d))
                print("#### Example Instance ####")
            
            builder.add_np_item(d)
            selected_num += 1
            
            pbar.update(1)
            
            if len(sorted_selected_indices) == 0:
                break
            idx = sorted_selected_indices.pop()

    builder.finalize()
        
    print(f"Selected Data Num: {selected_num}")


if __name__ == "__main__":
    main()