import multiprocessing
import os
import time
import torch
import sys
from numerize.numerize import numerize
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args
        
    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        line = line.replace("<@x(x!>", "\n")
        token_ids = Encoder.tokenizer.encode(line, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        
        return token_ids, len(line)


def main():
    args = get_args()
        
    args.processed_data_dir = os.path.join(args.processed_data_dir, numerize(args.train_num))

    os.makedirs(args.processed_data_dir, exist_ok=True)
        
    file_name = os.path.join(args.data_dir, "data.txt")
    fin = open(file_name, "r", encoding="utf-8")
    # encoder use the tokenizer to encode data
    encoder = Encoder(args)

    # 2. Mapping all datas with Encoder, with the help of multiprocessing
    pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
    encoded_docs = pool.imap_unordered(encoder.encode, fin, chunksize=50)
    proc_start = time.time()
    total_bytes_processed = 0

    # 3. tool `indexed_dataset` compress the tokenized data into binary format `bin_file`
    # it will also generate another small `idx_file` for saving meta information in order to decode `bin_file`.
    train_bin_file = os.path.join(args.processed_data_dir, f"train_{0}.bin")
    train_idx_file = os.path.join(args.processed_data_dir, f"train_{0}.idx")

    valid_bin_file = os.path.join(args.processed_data_dir, f"valid_{0}.bin")
    valid_idx_file = os.path.join(args.processed_data_dir, f"valid_{0}.idx")

    if args.model_type!="qwen":
        train_binary_builder = make_builder(train_bin_file, impl="mmap", dtype=np.uint16)
        valid_binary_builder = make_builder(valid_bin_file, impl="mmap", dtype=np.uint16)
    else:
        train_binary_builder = make_builder(train_bin_file, impl="mmap", dtype=np.uint32)
        valid_binary_builder = make_builder(valid_bin_file, impl="mmap", dtype=np.uint32)

    # put tokenized data into binary_builder
    buffer = []
    inst_num = 0
    for lid, (input_ids, bytes_processed) in enumerate(encoded_docs):
        total_bytes_processed += bytes_processed
        if input_ids is None:
            continue
        
        buffer.extend(input_ids)
        while len(buffer) >= args.max_length:
            inst = buffer[:args.max_length]
            buffer = buffer[args.max_length:]
        
            if inst_num < args.dev_num:
                valid_binary_builder.add_item(torch.IntTensor(inst))
            else:
                train_binary_builder.add_item(torch.IntTensor(inst))
            
            inst_num += 1
            
        if lid % 10000 == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {lid} documents. {inst_num} instances.",
                f"({lid/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)
        
        if inst_num - args.dev_num >= args.train_num:
            break

    # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
    train_binary_builder.finalize(train_idx_file)
    valid_binary_builder.finalize(valid_idx_file)

    # close multiproceessing mapping
    pool.close()


if __name__ == '__main__':
    main()