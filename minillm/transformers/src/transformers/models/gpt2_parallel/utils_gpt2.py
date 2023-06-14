import torch
import argparse
import os
import tqdm
import copy


column_weight = [
    "lm_head.weight",
    "wte.weight",
    "attn.c_attn.weight",
    "mlp.c_fc.weight",
]

row_weight = [
    "attn.c_proj.weight",
    "mlp.c_proj.weight",
]

column_bias = [
    "attn.c_attn.bias",
    "mlp.c_fc.bias",
]

no_mp_weights = [
    "wpe.weight",
    "ln_1",
    "ln_2",
    "masked_bias",
    "ln_f",
    "attn.bias",
    "mlp.c_proj.bias",
    "attn.c_proj.bias",
]


def increase_mp_gpt2(d, mp_size, half=False):

    print("Increase MP size.")

    ratio = mp_size
    start = 0
    end = ratio

    ckpts = []

    for j in tqdm.tqdm(range(start, end)):
        d_new = {}
        shift = j - start

        for k, v in d.items():
            print(k)
            if any([kk in k for kk in column_weight]):
                assert len(v.shape) < 3
                if "attn.c_attn" in k:
                    v = v.transpose(0, 1)
                    dim = v.shape[0] // 3
                    part = dim // ratio
                    d_new[k] = torch.cat(
                        [
                            v[shift*part:(shift+1)*part, :],
                            v[dim+shift*part:dim+(shift+1)*part, :],
                            v[2*dim+shift*part:2*dim+(shift+1)*part, :],
                        ],
                        dim=0
                    ).clone()
                elif "mlp.c_fc" in k:
                    v = v.transpose(0, 1)
                    part = v.shape[0] // ratio
                    d_new[k] = v[shift*part:(shift+1)*part, :].clone()
                else:
                    part = v.shape[0] // ratio
                    d_new[k] = v[shift*part:(shift+1)*part, :].clone()
            elif any([kk in k for kk in row_weight]):
                assert len(v.shape) < 3
                v = v.transpose(0, 1)
                part = v.shape[1] // ratio
                d_new[k] = v[:, shift*part:(shift+1)*part].clone()
            elif any([kk in k for kk in column_bias]):
                assert len(v.shape) < 3
                if "attn.c_attn" in k:
                    dim = v.shape[0] // 3
                    part = dim // ratio
                    d_new[k] = torch.cat(
                        [
                            v[shift*part:(shift+1)*part],
                            v[dim+shift*part:dim+(shift+1)*part],
                            v[2*dim+shift*part:2*dim+(shift+1)*part],
                        ],
                        dim=0
                    ).clone()
                else:
                    part = v.shape[0] // ratio
                    d_new[k] = v[shift*part:(shift+1)*part].clone()
            else:
                assert any([kk in k for kk in no_mp_weights]), k
                d_new[k] = v.clone()
                
            # print(k, d[k].size(), d[k].dtype, d_new[k].size(), d_new[k].dtype)
        
        ckpts.append(d_new)
        
    return ckpts


def decrease_mp_gpt2(d_list, half=False):

    print("Decrease MP size to 1.")

    d_new = {}
    
    for k, v in d_list[0].items():
        assert len(v.shape) < 3
        if any([kk in k for kk in column_weight]):
            d_new[k] = torch.cat([d[k] for d in d_list], dim=0)
        elif any([kk in k for kk in row_weight]):
            d_new[k] = torch.cat([d[k] for d in d_list], dim=1)
        elif any([kk in k for kk in column_bias]):
            d_new[k] = torch.cat([d[k] for d in d_list], dim=0)
        else:
            assert any([kk in k for kk in no_mp_weights]), k
            d_new[k] = v.clone()

    return d_new