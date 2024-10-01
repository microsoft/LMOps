import os
import time
import numpy as np
import argparse
import multiprocessing as mp

from utils import BOS_MODELS, get_tokenizer
from data_utils import ChunkedDatasetBuilder, best_fitting_dtype, DistributedMMapIndexedDataset
from arguments import add_data_args, add_runtime_args, add_hp_args, add_model_args


class Encoder(object): 
    def __init__(self, args):
        self.args = args
        self.old_model_type = args.old_model_type
        self.old_model_path = args.old_model_path
        self.new_model_type = args.model_type
        self.new_model_path = args.model_path

    def initializer(self):
        Encoder.tokenizer_old = get_tokenizer(
            self.args, model_path=self.old_model_path, model_type=self.old_model_type)
        Encoder.tokenizer_new = get_tokenizer(
            self.args, model_path=self.new_model_path, model_type=self.new_model_type)

    def encode(self, id_with_d):
        did, d = id_with_d
        d = d.astype(int)
        eos_poses = np.where(d == Encoder.tokenizer_old.eos_token_id)[0]
        start = 0
        split_d = []
        for p in eos_poses:
            split_d.append(d[start:p])
            start = p + 1
        split_d.append(d[start:])
        tokens = []
        for _d in split_d:
            _s = Encoder.tokenizer_old.decode(_d, skip_special_tokens=True)
            _tokens = Encoder.tokenizer_new.encode(_s, add_special_tokens=False)
            tokens.extend(_tokens)
            tokens.append(Encoder.tokenizer_new.eos_token_id)
        tokens.pop() # pop the last eos_token_id
        if self.args.model_type in BOS_MODELS:
            tokens = [Encoder.tokenizer_new.bos_token_id] + tokens[:self.args.max_length-1]

        return did, d, tokens, len(d)


def print_and_save(s, output_path):
    print(s)
    with open(os.path.join(output_path, "log.txt"), "a") as f:
        f.write(s + "\n")


def get_additional_args(parser):
    parser.add_argument("--old-model-type", type=str, default=None)
    parser.add_argument("--old-model-path", type=str, default=None)
    return parser


def get_args():
    parser = argparse.ArgumentParser()

    parser = get_additional_args(add_hp_args(add_model_args(
        add_data_args(add_runtime_args(parser)))))
    args = parser.parse_args()

    return args


def main():
    args = get_args()
        
    sid = args.min_state * args.chunk_num_per_shard

    output_dir = os.path.join(args.save, args.data_name, f"{args.old_model_type}-{args.model_type}-{args.max_length}")
    os.makedirs(output_dir, exist_ok=True)

    old_tokenizer = get_tokenizer(args, model_path=args.old_model_path, model_type=args.old_model_type)
    new_tokenizer = get_tokenizer(args, model_path=args.model_path, model_type=args.model_type)

    dtype = best_fitting_dtype(new_tokenizer.vocab_size)
    builder = ChunkedDatasetBuilder(
        args.base_path, output_dir, dtype,
        chunk_num_per_shard=args.chunk_num_per_shard,
        output_start_state=args.min_state)

    data = DistributedMMapIndexedDataset(args.data_dir, "data", min_state=args.min_state)
    encoder = Encoder(args)
    pool = mp.Pool(processes=args.data_process_workers,
                   initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, enumerate(data), chunksize=50)

    proc_start = time.time()
    total_bytes_processed = 0

    max_length_no_trunc = 0
    min_length_no_trunc = 1000000
    mean_length = 0

    for lid, (did, old_tokens, tokens, processed_bytes) in enumerate(encoded_docs):
        max_length_no_trunc = max(max_length_no_trunc, len(tokens))
        min_length_no_trunc = min(min_length_no_trunc, len(tokens))

        if lid == 0:
            print("#### Original tokens: ####")
            print(old_tokens, len(old_tokens))
            print(old_tokenizer.decode(old_tokens))
            print("#### New tokens: ####")
            print(tokens, len(tokens))
            print(new_tokenizer.decode(tokens))
        
        mean_length += len(tokens)
        total_bytes_processed += processed_bytes
        
        assert len(tokens) <= args.max_length
        
        sid += 1
        builder.add_np_item(np.array(tokens, dtype=dtype))

        if sid % 10000 == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print_and_save(f"Processed {sid} documents. " + 
                f"({lid/elapsed} docs/s, {mbs} MB/s).", output_dir)

    builder.finalize()

    pool.terminate()
    pool.close()
    pool.join()
    pool = None
    
    mean_length = mean_length / sid
    print_and_save(
        f"max_length_no_trunc: {max_length_no_trunc}, " + 
        f"min_length_no_trunc: {min_length_no_trunc}, " +
        f"mean_length: {mean_length}", output_dir)
        

if __name__ == "__main__":
    main()