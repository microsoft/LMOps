import os
import h5py
import numpy as np
import torch
import torch.distributed as dist
from utils import print_rank, all_gather


class Checkpointing():
    def __init__(self, args, tot_bs, total_steps, theta_size, device, rank):
        self.args = args
        self.tot_bs = tot_bs
        self.total_steps = total_steps
        self.theta_size = theta_size
        self.device = device
        self.r = rank
        self.data_cache_path = None
        self.model_cache_path = None
        self.dump_model_interval = int(np.sqrt(total_steps))
        self.ckpt_num = int(total_steps / self.dump_model_interval)
        self.dump_data_interval = self.dump_model_interval
        self.cached_data = []
        self.cached_micro_step_data = []
        self.dtype = torch.float32 if args.fp32 else torch.float16
        self.np_dtype = np.float32 if args.fp32 else np.float16
        
        self.setup_cache_paths()

    def print_setup(self):
        print_rank(f"dump_model_interval: {self.dump_model_interval}")
        print_rank(f"ckpt_num: {self.ckpt_num}")
        print_rank(f"dump_data_interval: {self.dump_data_interval}")

    def setup_cache_paths(self):
        self.data_cache_path = os.path.join(self.args.save, "data_cache.hdf5")
        self.model_cache_path = os.path.join(self.args.save, "model_cache.hdf5")

    def setup_h5(self, init_model):
        with h5py.File(self.data_cache_path, "w") as f:
            f.create_dataset("input_ids", (self.total_steps, self.tot_bs, self.args.max_length), dtype=(np.uint16), maxshape=(None, None, None), chunks=True)
            f.create_dataset("attention_mask", (self.total_steps, self.tot_bs, self.args.max_length), dtype=(np.int8), maxshape=(None, None, None), chunks=True)
            f.create_dataset("label", (self.total_steps, self.tot_bs, self.args.max_length), dtype=(np.uint16), maxshape=(None, None, None), chunks=True)
            f.create_dataset("loss_mask", (self.total_steps, self.tot_bs, self.args.max_length), dtype=(self.np_dtype), maxshape=(None, None, None), chunks=True)

        with h5py.File(self.model_cache_path, "w") as f:
            f.create_dataset("params", (self.ckpt_num + 1, self.theta_size), dtype=(self.np_dtype), maxshape=(None, None), chunks=True)
            f["params"][0] = init_model.get_params_vec().detach().cpu() # initial model

    def step(self):
        cached_micro_step_data = self.merge_cached_data(self.cached_micro_step_data)
        self.cached_data.append(cached_micro_step_data)
        self.cached_micro_step_data = []

    def append_micro_step_data(self, data):
        self.cached_micro_step_data.append(data)

    def dump_h5_data(self, data, step):
        with h5py.File(self.data_cache_path, "a") as f:
            f["input_ids"][step:step+data["input_ids"].size(0)] = data["input_ids"]
            f["attention_mask"][step:step+data["attention_mask"].size(0)] = data["attention_mask"]
            f["label"][step:step+data["label"].size(0)] = data["label"]
            f["loss_mask"][step:step+data["loss_mask"].size(0)] = data["loss_mask"]

    def load_h5_data(self, step):
        assert step >= 0
        data = {}
        if self.r == 0:
            with h5py.File(self.data_cache_path, "r") as f:
                data["input_ids"] = torch.tensor(f["input_ids"][step].astype(int), dtype=torch.long, device=self.device)
                data["attention_mask"] = torch.tensor(f["attention_mask"][step].astype(int), dtype=torch.long, device=self.device)
                data["label"] = torch.tensor(f["label"][step].astype(int), dtype=torch.long, device=self.device)
                data["loss_mask"] = torch.tensor(f["loss_mask"][step].astype(float), dtype=self.dtype, device=self.device)
        else:
            data["input_ids"] = torch.zeros(self.tot_bs, self.args.max_length, dtype=torch.long, device=self.device)
            data["attention_mask"] = torch.zeros(self.tot_bs, self.args.max_length, dtype=torch.long, device=self.device)
            data["label"] = torch.zeros(self.tot_bs, self.args.max_length, dtype=torch.long, device=self.device)
            data["loss_mask"] = torch.zeros(self.tot_bs, self.args.max_length, dtype=self.dtype, device=self.device)

        for k in data:
            dist.broadcast(data[k], 0)
        
        return data

    def load_h5_model(self, s):
        with h5py.File(self.model_cache_path, "r") as f:
            model_params_vec = f["params"][s]
        model_params_vec = torch.tensor(model_params_vec, device=self.device)
        return model_params_vec

    def dump_h5_model(self, model_params_vec, s):
        with h5py.File(self.model_cache_path, "a") as f:
            f["params"][s] = model_params_vec 

    def do_dump_model(self, global_steps):
        return global_steps % self.dump_model_interval == 0

    def dump_model(self, model, global_steps):
        # model
        if self.r == 0:
            model_params_vec = model.get_params_vec()
            self.dump_h5_model(model_params_vec.detach().cpu(), global_steps//self.dump_model_interval)
    
    def do_dump_data(self, global_steps):
        return global_steps % self.dump_data_interval == 0
    
    def dump_data(self, global_steps):
        # data
        if len(self.cached_data) == 0:
            return
        self.cached_data = self.merge_cached_data(self.cached_data)
        global_cached_data = {}
        for k, v in self.cached_data.items():
            global_v = all_gather(v, dim=3, op="stack")
            n, g, b, p, l = global_v.size() # n = act_ckpt_steps, g = gradient_acc_steps, p = ws, b = batch_size, l = max_length
            global_v = global_v.view(n, g * b * p, l)
            global_cached_data[k] = global_v.cpu()

        if self.r == 0:
            self.dump_h5_data(
                global_cached_data, global_steps-len(global_cached_data["input_ids"]))
        self.cached_data = []

    def merge_cached_data(self, cached_data):
        merged = {}
        for k in cached_data[0]:
            merged[k] = torch.stack([d[k] for d in cached_data], dim=0)
        return merged