import copy
import os
import math
import json
import traceback
from collections import defaultdict
from typing import List, Dict
from myloralib.config import ResLoraConfig
from myloralib.layers import ResLoraLinear
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F


from transformers import AutoConfig, PreTrainedModel, RobertaForSequenceClassification, LlamaForCausalLM, GPT2ForQuestionAnswering, LlamaConfig
from transformers.modeling_utils import unwrap_model

LORA_STATIC_MODEL = "pytorch_model_lora.bin"
LORA_STATIC_CONFIG = "reslora_config.json"

TARGET_MODULES_DICT = {
    "llama": {"q": ["q_proj"], "k": ["k_proj"], "v": ["v_proj"], "o": ["o_proj"], "f": ["up_proj", "down_proj"], "g": ["gate_proj"]},
    "unet": {"q": ["to_q"], "k": ["to_k"], "v": ["to_v"], "o": ["to_out"]},
    "mistral": {"q": ["q_proj"], "k": ["k_proj"], "v": ["v_proj"], "o": ["o_proj"], "f": ["up_proj", "down_proj"], "g": ["gate_proj"]},
    "roberta": {"q": ["query"], "k": ["key"], "v": ["value"], "o": ["out"]},
}

class ResLoraModel(nn.Module):
    def __init__(self, model: PreTrainedModel, resconfig: ResLoraConfig, epochs: int = -1, model_name="llama"):
        super().__init__()

        if resconfig.pre_num == -1:
            resconfig.pre_num = 999
        self.resconfig = resconfig
        model.config.resconfig = resconfig.to_json_string()
        self.config = model.config
        self.wrapped_model = model
        self.target_modules = set(resconfig.target_modules.lower().split("."))
        if model_name in TARGET_MODULES_DICT:
            cur_target_dict = TARGET_MODULES_DICT[model_name]
        else:
            raise NotImplementedError

        temp_target_modules = []
        for t in self.target_modules:
            if t in cur_target_dict:
                temp_target_modules += cur_target_dict[t]
            else:
                print(f"Warning: {t} is not in {model_name} target modules, please check your config.")
        self.ori_target_modules = self.target_modules
        self.target_modules = set(temp_target_modules)
        print(type(model))
        print("get target modules:", self.target_modules)
        self.lora_list = defaultdict(list)
        self.lora_name_list = defaultdict(list)

        for name, module in self.wrapped_model.named_modules():
            module.requires_grad_(False)
            if isinstance(module, nn.Linear):
                temp_flag = None
                for x in self.target_modules:
                    if x in name.lower():
                        temp_flag = x
                        break
                if temp_flag == None:
                    continue
                new_module = ResLoraLinear(module, resconfig, name)
                if self.weight_sample == None:
                    self.weight_sample = module.weight
                parent = self._get_parent(name)
                module_suffix = name.split(".")[-1]
                setattr(parent, module_suffix, new_module)
                self.lora_list[temp_flag].append(new_module)
                self.lora_name_list[temp_flag].append(name)

        if self.resconfig.res_flag > 0:
            print("all lora linear:", self.lora_name_list)
            self.concat_loras()


    def concat_loras(self):
        if self.resconfig.res_flag > 0:
            for k, v in self.lora_list.items():
                if len(v) > 1:
                    print(f"start to link {len(v)} lora blocks in {k}")
                    v[0].number = 0
                    for i in range(1, len(v)):
                        v[i].number = i
                        if v[i].in_features == v[i-1].in_features and v[i].out_features == v[i-1].out_features:
                            v[i].pre_lora = v[i-1]
                        else:
                            print(f"Warning: lora block {i} in {k} can not link to {i-1} because of different shape ({v[i].in_features}, {v[i].out_features}) and ({v[i-1].in_features}, {v[i-1].out_features}).")


    def unconcat_loras(self):
        if self.resconfig.res_flag > 0:
            for k, v in self.lora_list.items():
                if len(v) > 1:
                    print(f"start to unlink {len(v)} lora layers in {k}")
                    for i in range(1, len(v)):
                        v[i].pre_lora = None


    def calculate_froc(self):
        if self.resconfig.res_flag == 1 or self.resconfig.res_flag == 3:
            print("init froc and alpha")
            for k, v in self.lora_list.items():
                for lora in v:
                    lora.froc = -1
                    lora.alpha = -1
            print("start to calculate froc")
            for k, v in self.lora_list.items():
                for lora in v:
                    lora.calculate_froc()
                    assert lora.froc > 0, f"Error: lora {lora.name} froc is {lora.froc}"
                    
            for k, v in self.lora_list.items():
                for lora in v:
                    lora.calculate_alpha()
                    print(f"lora {lora.name}, froc: {lora.froc}, alpha: {lora.alpha}")
                    assert lora.alpha >= 0, f"Error: lora {lora.name} alpha is {lora.alpha}"
        else:
            raise NotImplementedError


    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent


    def generate(self, **kwargs):
        return self.wrapped_model.generate(**kwargs)


    def save_checkpoint(self, save_dir):
        print("saving on", save_dir)
        if self.resconfig.rank > 0:
            res = {}
            for name, module in self.wrapped_model.named_modules():
                if isinstance(module, self.lora_type):
                    for i in range(module.cur_lora_index+1):
                        res[f"{name}.loras.{i}.lora_A"] = copy.deepcopy(module.loras[i].lora_A)
                        res[f"{name}.loras.{i}.lora_B"] = copy.deepcopy(module.loras[i].lora_B)
            with open(f"{save_dir}/{LORA_STATIC_MODEL}", "wb") as f:
                torch.save(res, f)
            with open(f"{save_dir}/{LORA_STATIC_CONFIG}", "w") as f:
                json.dump(self.resconfig.__dict__, f, indent=4)
        else:
            self.wrapped_model.save_pretrained(save_dir)


    @staticmethod
    def load_checkpoint(model_dir: str, lora_dir=""):
        if "llama" in model_dir.lower():
            model = LlamaForCausalLM.from_pretrained(model_dir)
        elif "roberta" in model_dir.lower():
            model = RobertaForSequenceClassification(model_dir)
        elif "gpt" in model_dir.lower():
            model = GPT2ForQuestionAnswering(model_dir)
        else:
            raise NotImplementedError
        if lora_dir != "":
            with open(f"{lora_dir}/{LORA_STATIC_MODEL}", "rb") as f:
                reslora = torch.load(f)
            with open(f"{lora_dir}/{LORA_STATIC_CONFIG}", "r") as f:
                resconfig = json.load(f)
            resconfig = ResLoraConfig(**resconfig)
            model = ResLoraModel(model, resconfig)
            missing, unexpected = model.wrapped_model.load_state_dict(reslora, strict=False)
            if resconfig.start_index < resconfig.lora_num - 1:
                model.new_epoch()
            else:
                print("Warning: this ckpt can not change to new lora for lora_num and start_index in config.")
            print("Unexpected module:", unexpected)
            print("Miss module:", missing)
        return model


    @staticmethod
    def load_full_checkpoint(model_dir: str):
        model = LlamaForCausalLM(LlamaConfig.from_pretrained(model_dir))
        with open(f"{model_dir}/config.json", "r") as f:
            resconfig = ResLoraConfig(**json.load(f)["resconfig"])
        for k, v in resconfig.__dict__.items():
            print(k, v, type(v))
        model = ResLoraModel(model, resconfig)
        missing, unexpected = model.wrapped_model.from_pretrained(model_dir)
        if resconfig.start_index < resconfig.lora_num - 1:
            model.new_epoch()
        else:
            print("Warning: this ckpt can not change to new lora for lora_num and start_index in config.")
        print("Unexpected module:", unexpected)
        print("Miss module:", missing)
        return model


    def save_lora_checkpoint(self, save_dir):
        print("saving on", save_dir)
        if self.resconfig.rank > 0:
            res = {}
            for name, module in self.wrapped_model.named_modules():
                if isinstance(module, self.lora_type):
                    
                    lora_A, lora_B = deepcopy(module.loras[0].lora_A), deepcopy(module.loras[0].lora_B)
                    lora_A.requires_grad = False
                    lora_B.requires_grad = False
                    for lora in module.loras[1:(module.cur_lora_index+1)]:
                        lora_A += lora.lora_A
                        lora_B += lora.lora_B
                    res[f"{name}.loras.0.lora_A"] = lora_A
                    res[f"{name}.loras.0.lora_B"] = lora_B
            with open(f"{save_dir}/{LORA_STATIC_MODEL}", "wb") as f:
                torch.save(res, f)
            with open(f"{save_dir}/{LORA_STATIC_CONFIG}", "w") as f:
                json.dump(self.resconfig.__dict__, f, indent=4)
        else:
            self.wrapped_model.save_pretrained(save_dir)


    @staticmethod
    def load_lora_checkpoint(model_dir, lora_dir=""):
        if "llama" in model_dir.lower():
            model = LlamaForCausalLM.from_pretrained(model_dir)
        elif "roberta" in model_dir.lower():
            model = RobertaForSequenceClassification(model_dir)
        elif "gpt" in model_dir.lower():
            model = GPT2ForQuestionAnswering(model_dir)
        else:
            raise NotImplementedError
        if lora_dir != "":
            with open(f"{lora_dir}/{LORA_STATIC_MODEL}", "rb") as f:
                reslora = torch.load(f)
            with open(f"{lora_dir}/{LORA_STATIC_CONFIG}", "r") as f:
                resconfig = json.load(f)
            resconfig = ResLoraConfig(**resconfig)
            model = ResLoraModel(model, resconfig)
            model.new_epoch()
            missing, unexpected = model.wrapped_model.load_state_dict(reslora, strict=False)
            print("unexpected:", unexpected)
        return model


    def new_epoch(self, sche = None):
        index = -100
        self.step += 1
        for name, module in self.wrapped_model.named_modules():
            if isinstance(module, self.lora_type):
                index = module.change_lora()
        self.resconfig.start_index = index
        return index


    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.wrapped_model.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)


    def __call__(self, *args, **kwargs):
        return self.wrapped_model(*args, **kwargs)
