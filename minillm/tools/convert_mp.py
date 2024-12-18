import torch
import argparse
import os
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig


def increase_mp(d: dict[str, torch.Tensor], mp_size: int, mp_weights_config: dict):

    print("Increase MP size.")

    ratio = mp_size
    start = 0
    end = ratio

    ckpts = []

    for j in tqdm(range(start, end)):
        d_new = {}
        shift = j - start

        for k, v in d.items():
            assert len(v.shape) < 3
            if any([kk in k for kk in mp_weights_config["column_weight"]]):
                part = v.shape[0] // ratio
                d_new[k] = v[shift*part:(shift+1)*part, :].clone()
            elif any([kk in k for kk in mp_weights_config["row_weight"]]):
                part = v.shape[1] // ratio
                d_new[k] = v[:, shift*part:(shift+1)*part].clone()
            elif any([kk in k for kk in mp_weights_config["column_bias"]]):
                d_new[k] = v[shift*part:(shift+1)*part].clone()
            else:
                assert any([kk in k for kk in mp_weights_config["no_mp_weights"]]), k
                d_new[k] = v.clone()
        
        ckpts.append(d_new)
        
    return ckpts


def decrease_mp(d_list: list, mp_weights_config: dict):

    print("Decrease MP size to 1.")

    d_new = {}
    
    for k, v in d_list[0].items():
        assert len(v.shape) < 3
        if any([kk in k for kk in mp_weights_config["column_weight"]]):
            d_new[k] = torch.cat([d[k] for d in d_list], dim=0)
        elif any([kk in k for kk in mp_weights_config["row_weight"]]):
            d_new[k] = torch.cat([d[k] for d in d_list], dim=1)
        elif any([kk in k for kk in mp_weights_config["column_bias"]]):
            d_new[k] = torch.cat([d[k] for d in d_list], dim=0)
        else:
            assert any([kk in k for kk in mp_weights_config["no_mp_weights"]]), k
            d_new[k] = v.clone()

    return d_new


def main():
    parser = argparse.ArgumentParser("Change the tensor parallel of a model.")

    parser.add_argument("--input_path", type=str)
    parser.add_argument("--model_type", type=str, default="opt", choices=["opt", "qwen2", "llama", "mistral"])
    parser.add_argument("--source_mp_size", type=int, default=1)
    parser.add_argument("--target_mp_size", type=int, default=2)
    # parser.add_argument("--save_path", type=str)
    parser.add_argument("--dtype", choices=["torch.float32", "torch.float16", "torch.bfloat16"], default="torch.float16")
    parser.add_argument("--exist_ok", action="store_true")

    args = parser.parse_args()
    
    with open(os.path.join(os.path.dirname(__file__), f"mp_weights_configs/{args.model_type}.json"), "r") as f:
        mp_weights_config = json.load(f)
    
    if args.source_mp_size == 1:
        assert args.target_mp_size > args.source_mp_size
        args.save_path = os.path.join(args.input_path, f"mp{args.target_mp_size}")
        assert args.exist_ok or not any([os.path.exists(os.path.join(args.save_path, f"pytorch_model_{i}.bin")) for i in range(args.target_mp_size)])
        os.makedirs(args.save_path, exist_ok=True)
        model_hf = AutoModelForCausalLM.from_pretrained(args.input_path, torch_dtype=eval(args.dtype)).state_dict()
        d_list = increase_mp(model_hf, args.target_mp_size, mp_weights_config=mp_weights_config)
        for i, d in enumerate(d_list):
            torch.save(d, os.path.join(args.save_path, f"pytorch_model_{i}.bin"))
    elif args.target_mp_size == 1:
        assert args.source_mp_size > args.target_mp_size
        args.save_path = args.input_path
        assert args.exist_ok or not os.path.exists(os.path.join(args.save_path, "pytorch_model.bin"))
        ckpt_path = os.path.join(args.input_path, f"mp{args.source_mp_size}")
        d_list = [torch.load(os.path.join(ckpt_path, f"pytorch_model_{i}.bin"), map_location="cpu") for i in range(args.source_mp_size)]
        d = decrease_mp(d_list, mp_weights_config=mp_weights_config)
        torch.save(d, os.path.join(args.save_path, "pytorch_model.bin"))
        config = AutoConfig.from_pretrained(args.input_path)
        if hasattr(config, "is_model_parallel"):
            del config.is_model_parallel
    else:
        args.save_path = os.path.join(args.input_path, f"mp{args.target_mp_size}")
        assert args.exist_ok or not any([os.path.exists(os.path.join(args.save_path, f"pytorch_model_{i}.bin")) for i in range(args.target_mp_size)])
        
        ckpt_path = os.path.join(args.input_path, f"mp{args.source_mp_size}")
        d_list = [torch.load(os.path.join(ckpt_path, f"pytorch_model_{i}.bin"), map_location="cpu") for i in range(args.source_mp_size)]
        d = decrease_mp(d_list, mp_weights_config=mp_weights_config)
        
        torch.save(d, os.path.join(args.input_path, "pytorch_model.bin"))
        
        os.makedirs(args.save_path, exist_ok=True)
        model_hf = AutoModelForCausalLM.from_pretrained(args.input_path, torch_dtype=eval(args.dtype)).state_dict()
        d_list = increase_mp(model_hf, args.target_mp_size, mp_weights_config=mp_weights_config)
        for i, d in enumerate(d_list):
            torch.save(d, os.path.join(args.save_path, f"pytorch_model_{i}.bin"))
        

if __name__ == '__main__':
    main()
