"""Processing data"""
import os
import json
import time
import torch
import random
import argparse
import numpy as np
import multiprocessing

from utils import BOS_MODELS, get_tokenizer
from data_utils import ChunkedDatasetBuilder, best_fitting_dtype
from arguments import add_data_args, add_runtime_args, add_hp_args, add_model_args


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self,):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = get_tokenizer(self.args)

    def encode(self, id_with_json_line):
        doc_id, json_line = id_with_json_line
        line = json.loads(json_line)
        doc = line["text"]
        tokens = Encoder.tokenizer.encode(
            doc, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]

        return tokens, doc_id, len(doc)


def get_args():
    parser = argparse.ArgumentParser()

    parser = add_hp_args(add_model_args(
        add_data_args(add_runtime_args(parser))))
    args = parser.parse_args()

    return args


def check_sent_end(model_type, tokenizer, i, new_chunk, chunk_tokens_buffer):
    if model_type in BOS_MODELS:
        tokens = (new_chunk + chunk_tokens_buffer[1:2])[i:i+2]
    else:
        tokens = (new_chunk + chunk_tokens_buffer[:1])[i:i+2]
    s = tokenizer.decode(tokens)
    return len(tokens) == 1 or (" " in s)


def print_and_save(s, output_path):
    print(s)
    with open(os.path.join(output_path, "log.txt"), "a") as f:
        f.write(s + "\n")


def get_ent_sent_infos(args, tokenizer):
    with open(os.path.join(
        args.base_path, "tools", "process_data", f"end_sent_token_{args.model_type}.json"), "r") as f:
        end_sent_token = json.load(f)
    ent_sent_mask = np.zeros(tokenizer.vocab_size, dtype=np.uint8)
    for token in end_sent_token:
        ent_sent_mask[token] = True
    rt_token_mask = np.zeros(tokenizer.vocab_size, dtype=np.uint8)
    for token in end_sent_token:
        if "\n" in tokenizer.decode([token]):
            rt_token_mask[token] = True

    return ent_sent_mask, rt_token_mask


def main():
    args = get_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    g = torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = get_tokenizer(args)
    dtype = best_fitting_dtype(tokenizer.vocab_size)
        
    output_path = os.path.join(args.save, args.data_name, f"{args.model_type}-{args.max_length}")
    os.makedirs(output_path, exist_ok=True)
    print_and_save(f"Tokenizer vocab size: {tokenizer.vocab_size}. Using dtype: {dtype}", output_path)
    print_and_save(f"Input path: {args.data_dir} | Output path: {output_path}", output_path)
        
    with open(os.path.join(output_path, "args.json"), "w") as f:
        json.dump(vars(args), f)

    end_sent_mask, rt_token_mask = get_ent_sent_infos(args, tokenizer)

    builder = ChunkedDatasetBuilder(
        args.base_path,
        output_path,
        dtype,
        chunk_num_per_shard=args.chunk_num_per_shard,
        do_shuffle=True
    )

    startup_start = time.time()

    sid, lid = 0, 0
    max_chunk_length, min_chunk_length = 0, 1e9
    log_bytes_processed, log_doc_proccessed = 0, 0
    padded_token_num = 0
    if args.model_type in BOS_MODELS:
        chunk_tokens_buffer = [tokenizer.bos_token_id]
    else:
        chunk_tokens_buffer = []

    encoder = Encoder(args)
    pool = multiprocessing.Pool(args.data_process_workers, initializer=encoder.initializer)
    
    global_start = time.time()
    files_names =[]
    for _, _, files in os.walk(args.data_dir):
        for file_name in files:
            files_names.append(file_name)
            
    random.shuffle(files_names)
    
    with open(os.path.join(output_path, "file_names.json"), "w") as f:
        json.dump(files_names, f)
    
    proc_start = global_start
    
    for fid, file_name in enumerate(files_names):
        print_and_save(f"Processing {file_name}. {fid}/{len(files_names)}", output_path)
        input_file = os.path.join(args.data_dir, file_name)
        fin = open(input_file)

        # use the tokenizer to encode the sentences
        encoded_docs = pool.imap_unordered(encoder.encode, enumerate(fin), 50)

        for doc_tokens, doc_id, bytes_processed in encoded_docs:
            lid += 1
            log_bytes_processed += bytes_processed
            log_doc_proccessed += 1

            chunk_tokens_buffer.extend(doc_tokens)
            while len(chunk_tokens_buffer) >= args.max_length:
                new_chunk = chunk_tokens_buffer[:args.max_length]
                if args.model_type in BOS_MODELS:
                    chunk_tokens_buffer = [tokenizer.bos_token_id] + chunk_tokens_buffer[args.max_length:]
                else:
                    chunk_tokens_buffer = chunk_tokens_buffer[args.max_length:]
                for i in range(len(new_chunk)-1, -1, -1):
                    if (new_chunk[i] in [tokenizer.eos_token_id]) or \
                        (rt_token_mask[new_chunk[i]]) or \
                        (end_sent_mask[new_chunk[i]] and check_sent_end(args.model_type, tokenizer, i, new_chunk, chunk_tokens_buffer)):
                        # check if the end is fake
                        # 1. Who are you? I am the D.A. and he is //Bat Man. -> Who are you? // I am the D.A. and he is Bat Man.
                        # 2. Who are you? I am the D.//A. -> Who are you? // I am the D.A.
                        incomplete_sent = new_chunk[i+1:]
                        # new_chunk = new_chunk[:i+1] + [tokenizer.pad_token_id] * (args.max_length - (i+1))
                        new_chunk = new_chunk[:i+1]
                        if args.model_type in BOS_MODELS:
                            chunk_tokens_buffer = chunk_tokens_buffer[:1] + incomplete_sent + chunk_tokens_buffer[1:]
                        else:
                            chunk_tokens_buffer = incomplete_sent + chunk_tokens_buffer
                        padded_token_num += args.max_length - (i+1)

                        break
                
                if args.model_type in BOS_MODELS:
                    assert new_chunk[0] == tokenizer.bos_token_id
                if len(new_chunk) <= 1:
                    continue
                assert len(new_chunk) <= args.max_length

                sid += 1
                max_chunk_length = max(max_chunk_length, len(new_chunk))
                min_chunk_length = min(min_chunk_length, len(new_chunk))
                builder.add_np_item(np.array(new_chunk, dtype=dtype))

            if lid % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = log_bytes_processed / elapsed / 1024 / 1024
                ds = log_doc_proccessed / elapsed
                tokens = (sid * args.max_length - padded_token_num) / 1e9

                s = f"Processed {lid} documents. {sid} chunks. {tokens:.4f}B tokens. " + \
                    f"Padding fraction: {padded_token_num / (sid * args.max_length):.4f}. " + \
                    f"Max chunk length: {max_chunk_length}. Min chunk length: {min_chunk_length}. " + \
                    f"({ds:.2f} docs/s, {mbs:.4f} MB/s). Total Time: {current - global_start} s."

                print_and_save(s, output_path)

                log_bytes_processed, log_doc_proccessed = 0, 0
                proc_start = current

            if builder.ofid >= args.max_shard_num:
                break

        fin.close()
        fin = None
        
        if builder.ofid >= args.max_shard_num:
            break

    builder.finalize()

    if fin is not None:
        fin.close()

    pool.terminate()
    pool.close()
    pool.join()
    pool = None

    # summarize
    print_and_save(f"Total time: {time.time() - startup_start}.", output_path)
    print_and_save(f"Total processed paragraphs: {sid}.", output_path)
    total_tokens = sid * args.max_length - padded_token_num
    print_and_save(f"Total tokens: {total_tokens / 1e9:.4f}B", output_path)
    print_and_save(f"Total padding fraction: {padded_token_num / (sid * args.max_length)}.", output_path)


if __name__ == '__main__':
    main()
