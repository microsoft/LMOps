import json
import os
import numpy as np
import sys
import copy
import jsonlines
import time
import scipy.stats

task = sys.argv[1]
mode = sys.argv[2]
model = sys.argv[3]
model = f"en_dense_lm_{model}"

# !!! replace by your $base_dir/ana_rlt here
base_dir = "$base_dir/ana_rlt"
ana_rlt_dir = f"{base_dir}/{model}"

save_rlts = {}
debug_scale = 1
debug_n = 10000


def load_info(ana_setting):
    rlt_dir = f"{ana_rlt_dir}/{task}/{ana_setting}"
    info = []
    # jsonlines
    to_read_num = debug_n
    with open(f"{rlt_dir}/record_info.jsonl", "r") as f:
        for item in jsonlines.Reader(f):
            info.append(item)
            to_read_num -= 1
            if to_read_num == 0:
                break
    nfiles = len(info)
    print(rlt_dir, nfiles)
    info = info[:nfiles//debug_scale]
    return info


def cal_cos_sim(v1, v2):
    num = (v1 * v2).sum(axis=-1)  # dot product
    denom = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-20  # length
    res = num / denom
    return res


def cal_kl(v1, v2, axis=-1):
    res = scipy.stats.entropy(v1, v2, axis=-1)
    return res


def np_softmax(x, axis=-1):
    x -= np.max(x, axis=axis, keepdims=True)
    x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    return x


def check_answer(info_item):
    return info_item['gold_label'] == info_item['pred_label']


def norm_hidden(hidden):
    norm = np.linalg.norm(hidden, axis=-1) + 1e-20  # length
    norm = norm[:, :, np.newaxis]
    hidden = hidden / norm
    return hidden


stt_time = time.time()

ftzs_info = load_info('ftzs')
print(f'loading ftzs data costs {time.time() - stt_time} seconds')
stt_time = time.time()
zs_info = load_info('zs')
print(f'loading zs data costs {time.time() - stt_time} seconds')
stt_time = time.time()
icl_info = load_info('icl')
print(f'loading icl data costs {time.time() - stt_time} seconds')
stt_time = time.time()


def count_f2t():
    num_both_f2t = 0
    num_icl_f2t = 0
    num_ftzs_f2t = 0
    n_examples = len(zs_info)
    for i in range(n_examples):
        if not check_answer(zs_info[i]) and check_answer(icl_info[i]):
            num_icl_f2t += 1
        if not check_answer(zs_info[i]) and check_answer(ftzs_info[i]):
            num_ftzs_f2t += 1
        if not check_answer(zs_info[i]) and check_answer(ftzs_info[i]) and check_answer(icl_info[i]):
            num_both_f2t += 1

    print('================= number of both F2T examples:', num_both_f2t)
    print('================= number of ICL F2T examples:', num_icl_f2t)
    print('================= number of FTZS F2T examples:', num_ftzs_f2t)

    recall = num_both_f2t / num_ftzs_f2t * 100
    print('++++++++++++++++ ICL recall to FT:', f"{recall:.2f}")
    save_rlts['Recall2FTP'] = recall


def analyze_sim(mode, key, normalize=False):
    if mode == 'all':
        # ====================  check all hidden ======================
        zs_hidden = np.array([info_item[key] for info_item in zs_info])  # n_examples, n_layers, hidden_dim
        icl_hidden = np.array([info_item[key] for info_item in icl_info])  # n_examples, n_layers, hidden_dim
        ftzs_hidden = np.array([info_item[key] for info_item in ftzs_info])  # n_examples, n_layers, hidden_dim
    elif  mode == 'f2t':
        # ====================  check False->True hidden ======================
        zs_hidden = []
        icl_hidden = []
        ftzs_hidden = []
        n_examples = len(zs_info)
        for i in range(n_examples):
            if not check_answer(zs_info[i]) and check_answer(ftzs_info[i]) and check_answer(icl_info[i]):
                zs_hidden.append(zs_info[i][key])
                icl_hidden.append(icl_info[i][key])
                ftzs_hidden.append(ftzs_info[i][key])
        zs_hidden = np.array(zs_hidden)  # n_examples, n_layers, hidden_dim
        icl_hidden = np.array(icl_hidden)  # n_examples, n_layers, hidden_dim
        ftzs_hidden = np.array(ftzs_hidden)  # n_examples, n_layers, hidden_dim

    if normalize:
        zs_hidden = norm_hidden(zs_hidden)
        icl_hidden = norm_hidden(icl_hidden)
        ftzs_hidden = norm_hidden(ftzs_hidden)

    icl_updates = icl_hidden - zs_hidden  # n_examples, n_layers, hidden_dim
    ftzs_updates = ftzs_hidden - zs_hidden

    print('======' * 5, f'analyzing {key}', '======' * 5)

    cos_sim = cal_cos_sim(icl_updates, ftzs_updates)
    cos_sim = cos_sim.mean(axis=0)
    save_rlts['SimAOU'] = cos_sim.tolist()
    print("per-layer updates sim (icl-zs)&(ftzs-zs):\n", np.around(cos_sim, 4))
    cos_sim = cos_sim.mean()
    print("overall updates sim (icl-zs)&(ftzs-zs):\n", np.around(cos_sim, 4))
    print()

    random_updates = np.random.random(ftzs_updates.shape)
    baseline_cos_sim = cal_cos_sim(icl_updates, random_updates)
    baseline_cos_sim = baseline_cos_sim.mean(axis=0)
    save_rlts['Random SimAOU'] = baseline_cos_sim.tolist()
    print("per-layer updates sim (icl-zs)&(random):\n", np.around(baseline_cos_sim, 4))
    baseline_cos_sim = baseline_cos_sim.mean()
    print("overall updates sim (icl-zs)&(random):\n", np.around(baseline_cos_sim, 4))
    print()


def analyze_attn_map(mode, key, softmax=False, sim_func=cal_cos_sim):
    if mode == 'all':
        # ====================  check all attn_map ======================
        zs_attn_map = [info_item[key] for info_item in zs_info]  # n_examples, n_layers, n_heads, len
        icl_attn_map = [info_item[key] for info_item in icl_info]  # n_examples, n_layers, n_heads, len
        ftzs_attn_map = [info_item[key] for info_item in ftzs_info]  # n_examples, n_layers, n_heads, len
    elif  mode == 'f2t':
        # ====================  check False->True attn_map ======================
        zs_attn_map = []
        icl_attn_map = []
        ftzs_attn_map = []
        n_examples = len(zs_info)
        for i in range(n_examples):
            if not check_answer(zs_info[i]) and check_answer(ftzs_info[i]) and check_answer(icl_info[i]):
                zs_attn_map.append(zs_info[i][key])
                icl_attn_map.append(icl_info[i][key])
                ftzs_attn_map.append(ftzs_info[i][key])
    zs_attn_map = copy.deepcopy(zs_attn_map)  # n_examples, n_layers, n_heads, len
    icl_attn_map = copy.deepcopy(icl_attn_map)  # n_examples, n_layers, n_heads, len
    ftzs_attn_map = copy.deepcopy(ftzs_attn_map)  # n_examples, n_layers, n_heads, len

    max_zs_len = -1
    n_examples = len(zs_attn_map)
    n_layers = len(zs_attn_map[0])
    n_heads = len(zs_attn_map[0][0])
    for i in range(n_examples):
        if len(zs_attn_map[i][0][0]) > max_zs_len:
            max_zs_len = len(zs_attn_map[i][0][0])
    pad_value = -1e20 if softmax else 0
    for i in range(n_examples):
        for j in range(n_layers):
            for k in range(n_heads):
                pad_len = max_zs_len - len(zs_attn_map[i][j][k])
                zs_attn_map[i][j][k] = [pad_value] * pad_len + zs_attn_map[i][j][k]
                icl_attn_map[i][j][k] = [pad_value] * pad_len + icl_attn_map[i][j][k]
                ftzs_attn_map[i][j][k] = [pad_value] * pad_len + ftzs_attn_map[i][j][k]

    zs_attn_map = np.array(zs_attn_map)  # n_examples, n_layers, n_heads, len
    icl_attn_map = np.array(icl_attn_map)  # n_examples, n_layers, n_heads, len
    ftzs_attn_map = np.array(ftzs_attn_map)  # n_examples, n_layers, n_heads, len
    if softmax:
        zs_attn_map = np_softmax(zs_attn_map, axis=-1)
        icl_attn_map = np_softmax(icl_attn_map, axis=-1)
        ftzs_attn_map = np_softmax(ftzs_attn_map, axis=-1)

    print('======' * 5, f'analyzing {key}, softmax={softmax}', '======' * 5)

    sim = sim_func(icl_attn_map, zs_attn_map)
    sim = sim.mean(axis=2)
    sim = sim.mean(axis=0)
    save_rlts['ZSL SimAM'] = sim.tolist()
    print("per-layer direct sim (icl)&(zs):\n", np.around(sim, 4))
    sim = sim.mean()
    print("overall direct sim (icl)&(zs):\n", np.around(sim, 4))
    print()

    sim = sim_func(icl_attn_map, ftzs_attn_map)
    sim = sim.mean(axis=2)
    sim = sim.mean(axis=0)
    save_rlts['SimAM'] = sim.tolist()
    print("per-layer direct sim (icl)&(ftzs):\n", np.around(sim, 4))
    sim = sim.mean()
    print("overall direct sim (icl)&(ftzs):\n", np.around(sim, 4))
    print()


stt_time = time.time()

count_f2t()
print(f'count_f2t costs {time.time() - stt_time} seconds')
stt_time = time.time()

analyze_sim(mode, 'self_attn_out_hiddens', normalize=True)
print(f'analyze_sim costs {time.time() - stt_time} seconds')
stt_time = time.time()

analyze_attn_map(mode, 'attn_map', softmax=False, sim_func=cal_cos_sim)
print(f'analyze_attn_map (w/o softmax) costs {time.time() - stt_time} seconds')
stt_time = time.time()

with open(f'{base_dir}/rlt_json/{task}-{model}.json', 'w') as f:
    json.dump(save_rlts, f, indent=2)

print(f'saving data costs {time.time() - stt_time} seconds')
stt_time = time.time()