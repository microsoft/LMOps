import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
from torch.nn.modules.module import T
from myloralib.config import ResLoraConfig


class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_para = lora_dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class LoraBlock(nn.Module, LoRALayer):
    def __init__(self, in_feature, out_feature, linear_weight: torch.Tensor, lora_config: ResLoraConfig, grad=True, res_flag: int = 0):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=lora_config.rank, lora_alpha=lora_config.lora_alpha, lora_dropout=lora_config.lora_dropout,
                           merge_weights=lora_config.merge_weights)
        
        self.scaling = self.lora_alpha / self.r
        self.lora_A = nn.Parameter(linear_weight.new_zeros((self.r, in_feature)))
        self.lora_B = nn.Parameter(linear_weight.new_zeros((out_feature, self.r)))

        self.requires_grad_(grad)
        self.res_flag = res_flag
        self.xa = None

    def load_lora(self, lora_a: torch.Tensor, lora_b: torch.Tensor):
        assert lora_a.shape == self.lora_A
        assert lora_b.shape == self.lora_B
        self.lora_A = lora_a.to(self.lora_A)
        self.lora_B = lora_a.to(self.lora_B)

    def forward(self, x: torch.Tensor, pre_xa: torch.Tensor = None):
        if self.r > 0:
            if self.res_flag:
                cur_xa = self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                self.xa = cur_xa
                if pre_xa is None:
                    return (cur_xa @ self.lora_B.transpose(0, 1)) * self.scaling
                else:
                    return ((cur_xa + pre_xa) @ self.lora_B.transpose(0, 1)) * self.scaling
            else:
                return (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
        else:
            return 0

    def merge(self):
        return self.lora_B @ self.lora_A * self.scaling

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)


class ResLoraLinear(nn.Linear):
    
    def __init__(
            self,
            old_module: nn.Linear,
            resconfig: ResLoraConfig,
            name: str,
            **kwargs
    ):
        nn.Linear.__init__(self, old_module.in_features, old_module.out_features, bias=(old_module.bias is not None), **kwargs)
        self.resconfig: ResLoraConfig = resconfig
        self.merged = False
        self.cur_lora_index = resconfig.start_index
        self.r = resconfig.rank
        self.loras = nn.ModuleList()
        self.fan_in_fan_out = False

        self.name = name
        self.pre_lora = None
        self.cur_input: torch.Tensor = None
        self.cur_output: torch.Tensor = None
        self.number = -1
        self.merge_4_win = []

        self.froc = -1
        self.alpha = -1

        for i in range(resconfig.lora_num):
            self.loras.append(LoraBlock(self.in_features, self.out_features, old_module.weight, resconfig, grad=True,
                                        res_flag=(self.resconfig.res_flag==3)))
        self.reset_parameters()
        self.weight = old_module.weight
        self.weight.requires_grad = False
        if old_module.bias is not None:
            self.bias = old_module.bias
            self.bias.requires_grad = False

    def change_lora(self):
        assert self.cur_lora_index + 1 < self.resconfig.lora_num, "Error: out of lora index!"
        if self.cur_lora_index == -100:
            self.cur_lora_index = 0
        else:
            self.cur_lora_index += 1
        self.loras[self.cur_lora_index].requires_grad_(True)
        return self.cur_lora_index

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'loras'):
            for i in range(self.resconfig.lora_num):
                self.loras[i].reset_parameters()

    @torch.no_grad()
    def calculate_froc(self):
        assert (self.resconfig.res_flag == 1 or self.resconfig.res_flag == 3) and self.resconfig.merge_flag > 2, "Error: calculate_froc is called in error!"
        if self.resconfig.res_flag == 1:
            if self.resconfig.merge_flag == 3:
                temp_merge = self.weight
                for i in range(self.cur_lora_index + 1):
                    temp_merge += self.loras[i].lora_A.transpose(0, 1) @ self.loras[i].lora_B.transpose(0, 1)
                self.froc = torch.norm(temp_merge, p='fro').item()
            elif self.resconfig.merge_flag == 4:
                if len(self.merge_4_win) == 0:
                    self.froc = 0
                else:
                    self.froc = np.mean(self.merge_4_win)
            else:
                raise Exception("Error: merge_flag is wrong!")
        elif self.resconfig.res_flag == 3:
            if self.resconfig.merge_flag == 3:
                if len(self.merge_4_win) == 0:
                    self.froc = 0
                else:
                    self.froc = np.mean(self.merge_4_win)
            elif self.resconfig.merge_flag == 4:
                if len(self.merge_4_win) == 0:
                    self.froc = 0
                else:
                    self.froc = np.mean(self.merge_4_win)
            else:
                raise Exception("Error: merge_flag is wrong!")
        else:
            raise Exception("Error: res_flag is wrong!")

    @torch.no_grad()
    def calculate_alpha(self):
        assert (self.resconfig.res_flag == 1 or self.resconfig.res_flag == 3) and self.resconfig.merge_flag > 0, "Error: calculate_alpha is called in error!"
        assert self.froc >= 0, f"Error: froc is {self.froc} in {self.name}!"
        if self.resconfig.res_flag == 1:
            if self.resconfig.merge_flag == 3:
                if self.pre_lora is None or self.pre_lora.pre_lora is None:
                    alpha = 1
                else:
                    alpha = self.pre_lora.pre_lora.froc / self.pre_lora.froc
            elif self.resconfig.merge_flag == 4:
                if self.pre_lora is None or self.froc == 0 or self.pre_lora.froc == 0:
                    alpha = 1
                else:
                    alpha = self.pre_lora.froc / self.froc
            else:
                raise Exception("Error: merge_flag is wrong!")
        elif self.resconfig.res_flag == 3:
            if self.resconfig.merge_flag == 3:
                alpha = 0
            elif self.resconfig.merge_flag == 4:
                temp_lora = self
                alpha = 0
                index = 0
                while temp_lora.pre_lora is not None and index < self.resconfig.pre_num:
                    temp_lora = temp_lora.pre_lora
                    if self.froc == 0 or temp_lora.froc == 0:
                        temp_alpha = 1
                    else:
                        temp_alpha = temp_lora.froc / self.froc
                    alpha += temp_alpha
                    index += 1
            else:
                raise Exception("Error: merge_flag is wrong!")
        else:
            raise Exception("Error: res_flag is wrong!")
        self.alpha = alpha


    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)
        if self.r > 0:
            if self.resconfig.res_flag == 1:
                if self.resconfig.merge_flag != 0 and not self.training:
                    assert self.alpha >= 0, f"Error: alpha is -1 in {self.name}!"
                    result += (1 + self.alpha) * self.loras[0](x)
                else:
                    if self.resconfig.merge_flag == 4 and ((temp_norm := torch.norm(x, p='fro').item()) > 0):
                        if len(self.merge_4_win) >= self.resconfig.merge_4_len:
                            self.merge_4_win.pop(0)
                        assert temp_norm > 0, f"Error: temp_norm is {temp_norm} in {self.name}!"
                        self.merge_4_win.append(temp_norm)
                    self.cur_input = x
                    if self.pre_lora is None:
                        pre_x = x
                    else:
                        pre_x = self.pre_lora.cur_input
                    for i in range(self.cur_lora_index+1):
                        result += self.loras[i](x + pre_x)
            elif self.resconfig.res_flag == 2:
                cur_lora = self
                for i in range(self.resconfig.pre_num+1):
                    for j in range(self.cur_lora_index + 1):
                        result += cur_lora.loras[j](x)
                    if cur_lora.pre_lora is None:
                        break
                    else:
                        cur_lora = cur_lora.pre_lora
            elif self.resconfig.res_flag == 3:
                if not self.training and self.resconfig.merge_flag != 0:
                    assert self.alpha >= 0, f"Error: alpha is -1 in {self.name}!"
                    if self.resconfig.merge_flag == 3:
                        for j in range(self.cur_lora_index + 1):
                            if self.pre_lora is None:
                                temp_xa = None
                            else:
                                temp_xa = torch.zeros_like(self.pre_lora.loras[0].xa)
                                cur_lora = self
                                for i in range(self.resconfig.pre_num):
                                    cur_lora = cur_lora.pre_lora
                                    if cur_lora is None:
                                        break
                                    temp_value = cur_lora.loras[j](x)
                                    assert cur_lora.loras[j].xa is not None, f"Error: xa is None in {cur_lora.name}, this block is {self.name} when merging!"
                                    if self.froc == 0 or cur_lora.froc == 0:
                                        temp_alpha = 1
                                    else:
                                        temp_alpha = cur_lora.froc / self.froc
                                    temp_xa += cur_lora.loras[j].xa.clone() * temp_alpha
                            result += self.loras[j](x, temp_xa)
                    elif self.resconfig.merge_flag == 4:
                        result += (1 + self.alpha) * self.loras[0](x)
                    else:
                        raise Exception("Error: merge_flag is wrong!")
                else:
                    for j in range(self.cur_lora_index + 1):
                        if self.pre_lora is None:
                            temp_xa = None
                        else:
                            temp_xa = torch.zeros_like(self.pre_lora.loras[0].xa)
                            cur_lora = self
                            for i in range(self.resconfig.pre_num):
                                cur_lora = cur_lora.pre_lora
                                if cur_lora is None:
                                    break
                                assert cur_lora.loras[j].xa is not None, f"Error: xa is None in {cur_lora.name}, this block is {self.name}!"
                                temp_xa += cur_lora.loras[j].xa
                        result += self.loras[j](x, temp_xa)
                    if self.resconfig.merge_flag == 4 and ((temp_norm := torch.norm(self.loras[0].xa, p='fro').item()) > 0):
                        if len(self.merge_4_win) >= self.resconfig.merge_4_len:
                            self.merge_4_win.pop(0)
                        assert temp_norm > 0, f"Error: temp_norm is {temp_norm} in {self.name}!"
                        self.merge_4_win.append(temp_norm)
                    elif self.resconfig.merge_flag == 3 and ((temp_norm := torch.norm(x, p='fro').item()) > 0):
                        if len(self.merge_4_win) >= self.resconfig.merge_4_len:
                            self.merge_4_win.pop(0)
                        assert temp_norm > 0, f"Error: temp_norm is {temp_norm} in {self.name}!"
                        self.merge_4_win.append(temp_norm)
            elif self.resconfig.res_flag == 0:
                for i in range(self.cur_lora_index+1):
                    result += self.loras[i](x)
            else:
                raise Exception("Error: res_flag is wrong!")
        return result



class Linear(nn.Linear, LoRALayer):
    
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

