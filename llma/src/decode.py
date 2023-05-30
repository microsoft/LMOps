from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import json
from collections import defaultdict
from tqdm import tqdm

import time
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--type", type=str, default="llma")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--append_docs", action="store_true")
    parser.add_argument("--input_data_fn", type=str, default="dev.top10.gpt.jsonl")
    parser.add_argument("--forced_decoding", action="store_true")
    args = parser.parse_args()
    return args

def get_tokenizer_and_model(llama_path):
    tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    model = LlamaForCausalLM.from_pretrained(llama_path)
    model.half()
    model.cuda()
    return tokenizer, model

def truncate(doc, tokenizer, max_tokens=1024):
    if max_tokens <= 0:
        return doc
    tokens = tokenizer.tokenize(doc)[:max_tokens]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    doc = tokenizer.decode(token_ids)
    return doc

def load_data(input_fn, tokenizer):
    s_list = []
    with open(input_fn) as fin:
        for line in fin:
            s = json.loads(line)
            if 'result' not in s.keys() or 'text' not in s['result'].keys() or len(s['result']['text'].strip()) == 0:
                continue
            for i, doc in enumerate(s['docs']):
                s['docs'][i] = truncate(doc, tokenizer, max_tokens=768)
            s_list.append(s)
    return s_list


def get_ngrams(tokens, n):
    ngram_list = []
    for i in range(len(tokens)-n+1):
        ngram = ' '.join(tokens[i:i+n])
        ngram_list.append(ngram)
    return ngram_list

def match_length(seq_a, seq_b):
    l = 0
    for i in range(min(len(seq_a), len(seq_b))):
        if seq_a[i] != seq_b[i]:
            break
        l += 1
    return l

def prepare_ngrams(s, n, tokenizer, max_n=0):
    if max_n < n:
        max_n = n
    docs = s['docs']
    if 'result' in s.keys():
        gtext = s['result']['text']
        gtokens = tokenizer.tokenize(gtext)
        g_ngrams_list = []
        for l in range(n, max_n+1):
            g_ngrams = get_ngrams(gtokens, l)
            g_ngrams_list.append([l,g_ngrams])
        g_ngrams_list.reverse()
    else:
        gtokens = None
        g_ngrams_list = None

    doc_list = [tokenizer.tokenize(x) for x in s['docs']]
    doc_token_id_list = [tokenizer.convert_tokens_to_ids(x) for x in doc_list]

    doc_ngrams_list = []

    for l in range(n, max_n+1):
        doc_ngrams = defaultdict(list)
        for i, doc_tokens in enumerate(doc_list):
            ngram_list = get_ngrams(doc_tokens, l)
            for j, ngram in enumerate(ngram_list):
                doc_ngrams[ngram].append((i,j))
        doc_ngrams_list.append([l,doc_ngrams])
    doc_ngrams_list.reverse()

    return {"target_ngrams": g_ngrams_list, "doc_list": doc_list, "doc_ngrams": doc_ngrams_list, "target_tokens": gtokens, "doc_token_id_list": doc_token_id_list}


def base_generate(model, tokenizer, input_ids, gen_texts_ids, trigger_N=0, block_K=0, forced_decoding=False, ngrams_cache=None):
    prepend_ids = input_ids.cuda()
    generate_ids = None
    past_key_values = None
    if forced_decoding:
        eos = torch.tensor([[tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
        gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)

    step = 0
    step_length = 1
    while True:
        with torch.no_grad():
            output = model(input_ids=prepend_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True)
            logits = output['logits']
            logits = logits[:,-1:,:]
            output_ids = torch.argmax(logits,dim=-1)
            if forced_decoding:
                output_ids = gen_texts_ids[:,step:step+step_length].to(output_ids.device)
            prepend_ids = output_ids
            if generate_ids is None:
                generate_ids = output_ids
            else:
                generate_ids = torch.concat([generate_ids, output_ids],dim=1)
            past_key_values = output['past_key_values']
            output_ids = output_ids.cpu().numpy()
            step += 1
            if output_ids[0][-1] == tokenizer.eos_token_id:
                break
    return generate_ids.cpu()

def match_prefix(g_ngrams_list, doc_ngrams_list, step):
    for g_ngrams, doc_ngrams in zip(g_ngrams_list, doc_ngrams_list):
        n = g_ngrams[0]
        g_ngrams = g_ngrams[1]
        doc_ngrams = doc_ngrams[1]
        if step < n:
            continue
        if g_ngrams[step-n] in doc_ngrams.keys():
            return n, g_ngrams, doc_ngrams
    return 0, None, None

def make_kv_tupe(input_kv, step_length, accepted_step_length):
    kv_list = []
    for kv in input_kv:
        l = kv.shape[2]
        kv_list.append(kv[:,:,:l-step_length+accepted_step_length,:])
    return tuple(kv_list)

def make_past_key_values(past_key_values, step_length, accepted_step_length):
    if step_length == accepted_step_length:
        return past_key_values
    pkv_list = []
    for kv in past_key_values:
        kv = make_kv_tupe(kv, step_length, accepted_step_length)
        pkv_list.append(kv)
    return tuple(pkv_list)

def llma_generate(model, tokenizer, input_ids, gen_texts_ids, trigger_N, block_K, forced_decoding=False, ngrams_cache=None):
    prepend_ids = input_ids.cuda()
    generate_ids = None
    past_key_values = None
    doc_ngrams_list = ngrams_cache["doc_ngrams"]
    doc_token_id_list = ngrams_cache["doc_token_id_list"]
    if forced_decoding:
        gtokens = ngrams_cache["target_tokens"]
        g_ngrams_list = ngrams_cache["target_ngrams"]
        eos = torch.tensor([[tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
        gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)
    else:
        gtokens = []
        g_ngrams_list = []
        for nlist in doc_ngrams_list:
            g_ngrams_list.append((nlist[0], []))

    step = 0
    while True:
        prefix_n, g_ngrams, doc_ngrams = match_prefix(g_ngrams_list, doc_ngrams_list, step)
        if prefix_n > 0:
            copy_mode = True
            trigger_ngram = g_ngrams[step-prefix_n]
            i, j = doc_ngrams[trigger_ngram][0]
            copied_ids = doc_token_id_list[i][j+prefix_n:j+prefix_n+block_K-1]
            step_length = 1+len(copied_ids)
            copied_ids = torch.tensor([copied_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
            prepend_ids = torch.concat([prepend_ids,copied_ids], dim=-1)
        else:
            step_length = 1
            copy_mode = False
        with torch.no_grad():
            output = model(input_ids=prepend_ids,
                    past_key_values=past_key_values,
                    return_dict=True,
                    use_cache=True)
            logits = output['logits'][:,-step_length:,:]
            output_ids = torch.argmax(logits,dim=-1)
            accepted_step_length = step_length
            past_key_values = output['past_key_values']
            if forced_decoding:
                output_ids = gen_texts_ids[:,step:step+step_length].to(output_ids.device)
            if copy_mode:
                iids = prepend_ids.cpu().numpy()[0]
                oids = output_ids.cpu().numpy()[0]
                real_output_ids = [oids[0]]
                for pos in range(1,len(oids)):
                    if oids[pos-1] == iids[pos] and oids[pos-1] != tokenizer.eos_token_id:
                        real_output_ids.append(oids[pos])
                    else:
                        break
                accepted_step_length = len(real_output_ids)
                past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length)
                if accepted_step_length < step_length:
                    output_ids = output_ids[:,:accepted_step_length]
            step += accepted_step_length
            prepend_ids = output_ids[:,-1:]
            if generate_ids is None:
                generate_ids = output_ids
            else:
                generate_ids = torch.concat([generate_ids, output_ids],dim=1)
            output_ids = output_ids.cpu().numpy()
            output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
            if not forced_decoding:
                gtokens += output_tokens
                for pos in range(len(g_ngrams_list)):
                    l = g_ngrams_list[pos][0]
                    g_ngrams_list[pos] = (l, get_ngrams(gtokens, l))
            if output_ids[0,-1] == tokenizer.eos_token_id:
                break
    return generate_ids.cpu()


def run_time_test(s_list, decoding_fn, model, tokenizer, trigger_N, block_K, append_docs=True, forced_decoding=False):
    for s in s_list:
        ngrams_cache = prepare_ngrams(s, trigger_N, tokenizer, max_n=5)
        if 'result' in s.keys() and 'text' in s['result']:
            gen_texts_ids = tokenizer(s['result']['text'], return_tensors="pt").input_ids[:,1:]
        else:
            gen_texts_ids = None
        s["ngrams_cache"] = ngrams_cache
        s["gen_texts_ids"] = gen_texts_ids
        query = s['query']
        if append_docs:
            docs = '\n'.join(s['docs'])
            prompt = f"docs:\n{docs}\nquery: {query}\nanswer:"
        else:
            prompt = query
        inputs = tokenizer(prompt, return_tensors="pt")
        s["inputs"] = inputs
    
    acc_time = 0
    total_length = 0
    total_start_time = time.time()
    for s in tqdm(s_list):
        start_time = time.time()
        inputs = s["inputs"]
        ngrams_cache = s["ngrams_cache"]
        gen_texts_ids = s["gen_texts_ids"]
    
        generate_ids = decoding_fn(model, tokenizer, inputs.input_ids, gen_texts_ids, trigger_N=trigger_N, block_K=block_K, forced_decoding=forced_decoding, ngrams_cache=ngrams_cache)
        total_length = generate_ids.shape[-1] + total_length
        end_time = time.time()
        s_time = end_time-start_time
        acc_time = s_time + acc_time
        generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        s["output"] = generated
        print(generated)
    total_end_time = time.time()
    total_time = total_end_time-total_start_time
    return total_time

def main():
    args = get_args()
    print(args)
    llama_path = args.model_path
    tokenizer, model = get_tokenizer_and_model(llama_path)
    input_fn = args.input_data_fn
    s_list = load_data(input_fn, tokenizer)
    if args.type == "base":
        print("baseline decoding")
        total_time = run_time_test(s_list, base_generate, model, tokenizer, 1, 1, append_docs=args.append_docs, forced_decoding=args.forced_decoding)
        print(total_time)
    else:
        print("llma decoding")
        trigger_N = args.n
        block_K = args.k
        print(f"n={trigger_N}, k={block_K}")
        total_time = run_time_test(s_list, llma_generate, model, tokenizer, trigger_N, block_K, append_docs=args.append_docs, forced_decoding=args.forced_decoding)
        print(total_time)
    
if __name__ == "__main__":
    main()
