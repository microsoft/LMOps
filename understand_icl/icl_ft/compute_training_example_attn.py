import json
import os
import numpy as np
import sys
import copy
import random
import jsonlines
import time
import scipy.stats

task = sys.argv[1]
model = sys.argv[2]
model = f"en_dense_lm_{model}"

# !!! replace by your $base_dir/ana_rlt here
base_dir = "$base_dir/ana_rlt"
ana_rlt_dir = f"{base_dir}/{model}"

debug_scale = 1
debug_n = 100
icl_k_dict = {
    'cb': 19, 
    'sst2': 32, 
    'sst5': 32, 
    'subj': 32, 
    'mr': 32, 
    'agnews': 22, 
}
icl_k = icl_k_dict[task]
save_rlts = {}


def load_info(ana_setting):
    rlt_dir = f"{ana_rlt_dir}/{task}/{ana_setting}"
    info = []
    # jsonlines
    to_read_num = 100000 if ana_setting == 'ft' else debug_n
    with open(f"{rlt_dir}/record_info.jsonl", "r") as f:
        for item in jsonlines.Reader(f):
            if ana_setting == 'ft' or ana_setting == 'ftzs':
                item = item['attn_q']
            elif ana_setting == 'icl':
                item = item['attn_map_ctx']
            else:
                assert 1 == 2
            info.append(item)
            to_read_num -= 1
            if to_read_num == 0:
                break
    n_records = len(info)
    print(rlt_dir, n_records)
    info = info[:n_records//debug_scale]
    return info


stt_time = time.time()

ft_attn_q = load_info('ft')  # (n_example, n_layers, ~len, q_hidden_dim)
print(f'loading ft data costs {time.time() - stt_time} seconds')
stt_time = time.time()

ftzs_attn_q = load_info('ftzs')  # (n_example, n_layers, q_hidden_dim)
print(f'loading ftzs data costs {time.time() - stt_time} seconds')
stt_time = time.time()

icl_attn_map_ctx = load_info('icl')  # (n_example, n_layers, n_head, tot_len)
print(f'loading icl data costs {time.time() - stt_time} seconds')
stt_time = time.time()


stored_slice_range = []


def slice_icl_attn_map(icl_attn_map_ctx):
    # (n_example, n_layers, n_head, tot_len)
    n_example = len(ft_attn_q)
    example_len = []
    for i in range(n_example):
        example_len.append(len(ft_attn_q[i][0]) - 1)

    slice_idx = []
    cur_pos = 1
    for i in range(n_example):
        new_range = list(range(cur_pos, cur_pos + example_len[i]))
        slice_idx.extend(new_range)
        cur_pos += example_len[i] + 1
    
    cur_pos = 0
    for i in range(n_example):
        new_range = list(range(cur_pos, cur_pos + example_len[i]))
        stored_slice_range.append(new_range)
        cur_pos += example_len[i] 

    icl_attn_map_ctx = icl_attn_map_ctx[:, :, slice_idx]
    return icl_attn_map_ctx


icl_attn_map_ctx = np.array(icl_attn_map_ctx)  # (n_example, n_layers, tot_len)
icl_attn_map_ctx = slice_icl_attn_map(icl_attn_map_ctx)
print('sliced icl_attn_map_ctx shape:', icl_attn_map_ctx.shape)
print(f'slicing icl_attn_map_ctx costs {time.time() - stt_time} seconds')
stt_time = time.time()


def compact_ft_attn_q(ft_attn_q):
    # (n_example, n_layers, ~len, q_hidden_dim)
    n_examples = len(ft_attn_q)
    n_layers = len(ft_attn_q[0])
    compacted_ft_attn_q = []
    for i in range(n_layers):
        layer_ft_attn_q = []
        for j in range(n_examples):
            cur_len = len(ft_attn_q[j][i])
            for k in range(1, cur_len):
                q_vector = ft_attn_q[j][i][k]
                layer_ft_attn_q.append(q_vector)
        compacted_ft_attn_q.append(layer_ft_attn_q)
    return compacted_ft_attn_q


ft_attn_q = compact_ft_attn_q(ft_attn_q)
ft_attn_q = np.array(ft_attn_q)  # (n_layers, tot_len, q_hidden_dim)
print('compacted ft_attn_q shape:', ft_attn_q.shape)
print(f'compacting ft_attn_q costs {time.time() - stt_time} seconds')
stt_time = time.time()


def cal_cos_sim(v1, v2):
    num = (v1 * v2).sum(axis=-1)  # dot product
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-20  # length
    res = num / denom
    return res


def norm_hidden(hidden):
    norm = np.linalg.norm(hidden, axis=-1) + 1e-20  # length
    norm = norm[:, :, np.newaxis]
    hidden = hidden / norm
    return hidden


def np_softmax(x, axis=-1):
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x


def compute_training_attn(level='token'):
    expand_ftzs_attn_q = np.array(ftzs_attn_q)[:, :, np.newaxis, :]  # (n_example, n_layers, 1, q_hidden_dim)
    expand_ft_attn_q = ft_attn_q[np.newaxis, :, :, :]  # (1, n_layers, tot_len, q_hidden_dim)
    
    ftzs_attn_map = (expand_ft_attn_q * expand_ftzs_attn_q).sum(axis=-1)  # (n_example, n_layers, tot_len)
    del expand_ftzs_attn_q
    del expand_ft_attn_q
    icl_attn_map = icl_attn_map_ctx  # (n_example, n_layers, tot_len)

    icl_attn_map = np_softmax(icl_attn_map, axis=-1)
    ftzs_attn_map = np_softmax(ftzs_attn_map, axis=-1)

    if level == 'example':
        n_example = len(stored_slice_range)
        tmp_ftzs_attn_map = np.random.random([ftzs_attn_map.shape[0], ftzs_attn_map.shape[1], n_example])
        tmp_icl_attn_map = np.random.random([ftzs_attn_map.shape[0], ftzs_attn_map.shape[1], n_example])
        for i, slice_range in enumerate(stored_slice_range):
            ftzs_attn_map_example = ftzs_attn_map[:, :, slice_range].sum(axis=-1)  # (n_example, n_layers)
            icl_attn_map_example = icl_attn_map[:, :, slice_range].sum(axis=-1)  # (n_example, n_layers)
            tmp_ftzs_attn_map[:, :, i] = ftzs_attn_map_example
            tmp_icl_attn_map[:, :, i] = icl_attn_map_example
        ftzs_attn_map = tmp_ftzs_attn_map
        icl_attn_map = tmp_icl_attn_map

    # =================== analyze recall of attention top-k ========================
    topk_k = 5 if level == 'example' else 100
    recall_topk = [0 for i in range(icl_attn_map.shape[1])]
    random_recall_topk = [0 for i in range(icl_attn_map.shape[1])]
    total_topk = 0
    for i in range(icl_attn_map.shape[0]):
        for j in range(icl_attn_map.shape[1]):
            ftzs_topk = np.argsort(-ftzs_attn_map[i][j])[:topk_k].tolist()
            icl_topk = np.argsort(-icl_attn_map[i][j])[:topk_k].tolist()
            for l in range(topk_k):
                if icl_topk[l] in ftzs_topk:
                    recall_topk[j] += 1
                random_idx = random.randint(0, icl_attn_map.shape[2] - 1)
                if random_idx in ftzs_topk:
                    random_recall_topk[j] += 1
            total_topk += topk_k
    total_topk /= icl_attn_map.shape[1]
    recall_topk = [recall_topk[i] / total_topk for i in range(icl_attn_map.shape[1])]
    random_recall_topk = [random_recall_topk[i] / total_topk for i in range(icl_attn_map.shape[1])]
    save_rlts[f'{level}-level recall_topk'] = recall_topk
    save_rlts[f'{level}-level random_recall_topk'] = random_recall_topk
    print('=========================================')
    print(f'{level}-level recall_topk:', recall_topk, ' ||| ', np.mean(recall_topk))
    print(f'{level}-level random_recall_topk:', random_recall_topk, ' ||| ', np.mean(random_recall_topk))
    
    # =================== analyze kendall of attention weights ========================
    
    kendall = [0 for i in range(len(icl_attn_map[0]))]
    random_kendall = [0 for i in range(len(icl_attn_map[0]))]
    for i in range(icl_attn_map.shape[0]):
        for j in range(icl_attn_map.shape[1]):
            cur_icl_map = np.array(icl_attn_map[i][j])
            cur_ftzs_map = np.array(ftzs_attn_map[i][j])
            random_map = np.random.random(ftzs_attn_map[i][j].shape)

            kendall[j] += scipy.stats.kendalltau(cur_icl_map, cur_ftzs_map)[0]
            random_kendall[j] += scipy.stats.kendalltau(cur_icl_map, random_map)[0]
    
    kendall = [kendall[i] / icl_attn_map.shape[0] for i in range(icl_attn_map.shape[1])]
    random_kendall = [random_kendall[i] / icl_attn_map.shape[0] for i in range(icl_attn_map.shape[1])]
    save_rlts[f'{level}-level kendall'] = kendall
    save_rlts[f'{level}-level random_kendall'] = random_kendall
    print('=========================================')
    print(f'{level}-level kendall:', kendall, ' ||| ', np.mean(kendall))
    print(f'{level}-level random_kendall:', random_kendall, ' ||| ', np.mean(random_kendall))


compute_training_attn(level='example')
compute_training_attn(level='token')
print(f'computing attention costs {time.time() - stt_time} seconds')
stt_time = time.time()
with open(f'{base_dir}/rlt_json/{task}-{model}_training_attn.json', 'w') as f:
    json.dump(save_rlts, f, indent=2)
print(f'saving data costs {time.time() - stt_time} seconds')
stt_time = time.time()