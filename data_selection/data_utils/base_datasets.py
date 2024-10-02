import os
from torch.utils.data import Dataset
from .distributed_indexed import DistributedMMapIndexedDataset

from torch.distributed import get_rank, get_world_size, is_initialized
from utils import print_rank
from tqdm import tqdm
import json
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, data_path=None, num=None, ada_max_length=False, data_name="", **kwargs):
        super().__init__()
        
        num_log = str(num) if num is not None else "ALL"
        print_rank(f"Load {split} from {data_path} with {num_log} instances")
        self.tokenizer = tokenizer
        self.args = args
        self.tokenizer = tokenizer
        self.split = split
        self.data_path = data_path
        self.num = num
        self.ada_max_length = ada_max_length
        self.data_name = data_name
        self.pad_id = self.tokenizer.pad_token_id
        self.eod_id = self.tokenizer.eos_token_id
        self.max_length = args.max_length
        self.min_prompt_length = args.min_prompt_length
        self.max_prompt_length = args.max_prompt_length
        self.answers = None
        self.order = None
        self.epoch = 0
        self.skip_offset = (-1, -1)

        assert data_path is not None
        self.load_data(**kwargs)
        
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            with open(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        elif os.path.exists(os.path.join(data_path, f"{split}.jsonl")):
            with open(os.path.join(data_path, f"{split}.jsonl")) as f:
                self.raw = [json.loads(line) for line in f.readlines()]
                self.answers = [x["output"] if isinstance(x["output"], list) else [x["output"]] for x in self.raw]
        else:
            print_rank("WARNING: No answers exist")
        
        if self.answers is not None:
            self.label_map = {tokenizer.encode(x[0], add_special_tokens=False)[0]: x[0] for x in self.answers}

        self.num = min(num, len(self.data)) if num is not None else len(self.data)
        assert self.num is not None and self.num > 0

        print_rank(f"Num instances: {self.num}")
            
    def __len__(self):
        return self.num

    def load_data(self, **kwargs):
        if self.args.bin_data:
            self.data = self.load_data_bin(self.data_path, **kwargs)
        elif self.args.json_data:
            self.data, self.origin_data = self.load_data_json(self.data_path)
        else:
            # txt data
            self.data = self.load_data_txt(self.data_path)

    def load_full_data(self):
        self.data = [np.array(self.data[i].astype(int).tolist()) for i in range(self.num)]

    def load_data_bin(self, data_path, **kwargs):
        r = get_rank() if is_initialized() else 0
        n = get_world_size() if is_initialized() else 1
        data = DistributedMMapIndexedDataset(data_path, f"{self.split}", r, n,
                                                    min_state=kwargs.get("min_state", 0), max_state=kwargs.get("max_state", None),
                                                    min_offset=kwargs.get("min_offset", 0), max_offset=kwargs.get("max_offset", None),
                                                    do_probe=kwargs.get("do_probe", True),
                                                    )        
        return data

    def load_data_json(self, data_path):
        if os.path.exists(os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")):
            data_path = os.path.join(data_path, f"{self.split}_{self.args.model_type}.jsonl")
        else:
            data_path = os.path.join(data_path, f"{self.split}.jsonl")
        
        with open(data_path) as f:
            lines = f.readlines()
        data_origin = [json.loads(line) for line in lines]
        data = []
        print_rank("Loading Data")
        for d in tqdm(data_origin, disable=(get_rank() != 0)):
            prompt = d["prompt"].replace("<n>", "\n")
            prompt_ids = self.tokenizer.encode(prompt)
            output_ids = None
            if "output" in d:
                if isinstance(d["output"], list):
                    output_ids = self.tokenizer.encode(d["output"][0])
                else:
                    output_ids = self.tokenizer.encode(d["output"])
            data.append({
                "prompt_ids": prompt_ids,
                "output_ids": output_ids[:self.max_length - self.max_prompt_length]
            })
        print_rank("Load End")
        return data, data_origin

    def load_data_txt(self, data_path):
        with open(os.path.join(data_path, f"{self.split}.txt")) as f:
            lines = f.readlines()
        data = []
        print_rank("Loading Data")
        for line in lines:
            line = line.strip()
            line = line.replace("<n>", "\n")
            prompt = self.tokenizer.encode(line)
            data.append(prompt)
        print_rank("Load End")
        return data

    def verbalizer(self):
        return self.label_map

    def set_order(self, path):
        self.order = np.load(path, mmap_mode="r")
        assert self.order.shape[1] <= self.num
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_num(self, num):
        self.num = num

    def set_skip_offset(self, skip_offset):
        self.skip_offset = tuple(skip_offset)

    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError

    def move_to_device(self, model_batch, no_model_batch, device):
        for k in model_batch:
            model_batch[k] = model_batch[k].to(device)   
             
        for k in no_model_batch:
            no_model_batch[k] = no_model_batch[k].to(device)    
        
        return model_batch, no_model_batch
