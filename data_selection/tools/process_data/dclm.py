import os
import sys
import json
import time
import numpy as np
import multiprocessing

from data_utils import ChunkedDatasetBuilder, best_fitting_dtype
from arguments import get_args
from utils import BOS_MODELS, get_tokenizer


class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        Encoder.tokenizer = get_tokenizer(self.args)

    def encode(self, line):
        line = json.loads(line)
        doc = line["text"]
        tokens = Encoder.tokenizer.encode(doc, add_special_tokens=False) + \
            [Encoder.tokenizer.eos_token_id]
        if self.args.model_type in BOS_MODELS:
            tokens = [Encoder.tokenizer.bos_token_id] + tokens        
        tokens = tokens[:self.args.max_length]        
        return tokens, doc, len(doc)


def main():
    args = get_args()
        
    output_dir = os.path.join(args.save, args.data_name, f"{args.model_type}-{args.max_length}")
    os.makedirs(output_dir, exist_ok=True)
        
    tokenizer = get_tokenizer(args)
    dtype = best_fitting_dtype(tokenizer.vocab_size)
    
    with open(os.path.join(args.data_dir, "sample_doc_10k.jsonl")) as f:
        lines = f.readlines()

    all_data = {
        "test": lines
    }
    
    print("Data num", len(lines))
    
    for split in all_data:
        
        encoder = Encoder(args)

        pool = multiprocessing.Pool(processes=args.data_process_workers, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(encoder.encode, all_data[split], chunksize=50)
        proc_start = time.time()
        total_bytes_processed = 0
        
        builder = ChunkedDatasetBuilder(args.base_path, output_dir, dtype, do_shuffle=True, split=split)
        
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        token_lens = []
        
        for lid, (tokens, doc, bytes_processed) in enumerate(encoded_docs):
            total_bytes_processed += bytes_processed
            
            if lid == 0:
                print("Doc:", tokenizer.decode(tokens))
                print(tokens)
                        
            builder.add_np_item(np.array(tokens, dtype=dtype))
            token_lens.append(len(tokens))   
            
            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)

        builder.finalize()
        pool.close()
                
        print("Data num", len(token_lens))
        print(f"Mean tokens len: {np.mean(token_lens)} | Max tokens len: {np.max(token_lens)} | Min tokens len: {np.min(token_lens)}")

if __name__ == '__main__':
    main()