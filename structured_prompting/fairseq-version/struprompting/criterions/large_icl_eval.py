# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from omegaconf import II
from omegaconf import open_dict

import torch
import copy
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from typing import Any, Dict, Iterator, List
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.distributed.utils import all_gather_list
from struprompting.data.squad.squad_metrics import compute_f1, compute_exact


@dataclass
class LargeICLEvalConfig(FairseqDataclass):
    is_generation: bool = field(
        default=False,
        metadata={
            "help": "open-ended genetation, e.g., squad"
        },
    )
    beam: int = field(
        default=3,
        metadata={
            "help": "beam size"
        },
    )


@register_criterion("large_icl_eval", dataclass=LargeICLEvalConfig)
class LargeICLEvalCriterion(FairseqCriterion):
    def __init__(self, cfg: LargeICLEvalConfig, task):
        super().__init__(task)
        self.cfg = cfg
        self.is_generation = cfg.is_generation
        self.valid_num_sum = 0
        self.acc_record = 0

        # remove repetition
        self.already_sample_num = 0
        self.src_tokens_prev = None
        self.sync_key_and_value = False

    def decode(self, tokens: torch.LongTensor):
        assert tokens.dim() == 1
        tokens = tokens[tokens!=1]
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.dictionary.bos():
            tokens = tokens[1:]  
        sentences = self.task.tokenizer.decode(self.task.dictionary.string(tokens))
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def forward(self, model, sample, reduce=True, split="valid"):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert split == "valid" or split == "demons"
        if model.decoder.embed_positions.max_pos_train <= 0:
            model.decoder.embed_positions.max_pos_train = self.task.max_pos_train
            model.decoder.demons_num = self.task.demons_num
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if split == "demons": 
            net_output, extra = model(
                **sample["net_input"],
                features_only=True
            )

            qkv_val = extra["qkv_val"]
            src_tokens = sample["net_input"]["src_tokens"]
            assert len(src_tokens) == 1
            local_src_tokens = sample["net_input"]["src_tokens"]
            all_local_src_tokens = [torch.zeros_like(local_src_tokens).to('cuda') for _ in range(world_size)]
            dist.all_gather(all_local_src_tokens, local_src_tokens)
            feature_mask = torch.logical_and(src_tokens != model.decoder.bos, src_tokens != model.decoder.pad)

            if self.src_tokens_prev is None:
                # first batch of demons
                model.decoder.context_keys, model.decoder.context_values = [], []
                for i in range(len(qkv_val)):
                    model.decoder.context_keys.append(qkv_val[i]["k"].transpose(0, 1)[feature_mask].to("cpu"))
                    model.decoder.context_values.append(qkv_val[i]["v"].transpose(0, 1)[feature_mask].to("cpu"))
            else:
                is_repetition = False
                if self.task.demons_num - self.already_sample_num < world_size:
                    for i in range(len(self.src_tokens_prev)):
                        if (local_src_tokens == self.src_tokens_prev[i]).all():
                            is_repetition = True
                            break
                
                if not is_repetition:
                    for i in range(len(qkv_val)):
                        model.decoder.context_keys[i] = torch.cat([model.decoder.context_keys[i], qkv_val[i]["k"].transpose(0, 1)[feature_mask].to("cpu")], 0)
                        model.decoder.context_values[i] = torch.cat([model.decoder.context_values[i], qkv_val[i]["v"].transpose(0, 1)[feature_mask].to("cpu")], 0)

            self.already_sample_num += world_size
            self.src_tokens_prev = all_local_src_tokens

            sample_size = sample["targets"].size(0)
            logging_output = {
                "loss": 0,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
            }
            return 0, sample_size, logging_output
        
        if not self.sync_key_and_value:
            local_size = torch.LongTensor(list(model.decoder.context_keys[0].size())[0:1]).to("cuda")
            size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
            dist.all_gather(size_list, local_size)
            size_list = [int(size.item()) for size in size_list]
            max_size = max(size_list)

            is_repetition = [False for i in range(world_size)]
            if self.task.demons_num < world_size:
                for i in range(1, world_size):
                    for j in range(0, i):
                        if (self.src_tokens_prev[i] == self.src_tokens_prev[j]).all():
                            is_repetition[i] = True

            layer_num = len(model.decoder.context_keys)
            for i in range(layer_num):
                # we pad the tensor because torch all_gather does not support
                # gathering tensors of different shapes
                all_context_keys = [torch.zeros_like(model.decoder.context_keys[i])[0:1].expand(max_size, -1).to('cuda') for _ in range(world_size)]
                all_context_values = [torch.zeros_like(model.decoder.context_values[i])[0:1].expand(max_size, -1).to('cuda') for _ in range(world_size)]

                local_keys = model.decoder.context_keys[i].clone().to('cuda')
                local_values = model.decoder.context_values[i].clone().to('cuda')

                padding_keys = torch.zeros_like(local_keys)[0:1].expand(max_size-len(local_keys), -1)
                padding_values = torch.zeros_like(local_values)[0:1].expand(max_size-len(local_values), -1)

                local_keys = torch.cat([local_keys, padding_keys], dim=0)
                local_values = torch.cat([local_values, padding_values], dim=0)

                dist.all_gather(all_context_keys, local_keys)
                dist.all_gather(all_context_values, local_values)

                # concat representations of all examples
                final_context_keys = None
                final_context_values = None
                for j in range(len(all_context_keys)):
                    if not is_repetition[j]:
                        if final_context_keys is None:
                            final_context_keys = all_context_keys[j][:size_list[j]]
                            final_context_values = all_context_values[j][:size_list[j]]
                        else:
                            final_context_keys = torch.cat([final_context_keys, all_context_keys[j][:size_list[j]]], dim=0)
                            final_context_values = torch.cat([final_context_values, all_context_values[j][:size_list[j]]], dim=0)

                model.decoder.context_keys[i] = final_context_keys.to('cpu')
                model.decoder.context_values[i] = final_context_values.to('cpu')
            self.sync_key_and_value = True

        if not self.is_generation:
            option_num = self.task.fewshot_task.class_num
            assert sample["targets"].size(0) % option_num == 0
            sample_size = sample["targets"].size(0) // option_num
            loss_mask = sample["net_input"]["gpt_loss_mask"][:, 1:]

            if len(loss_mask) == loss_mask.int().sum():
                # classification tasks
                shape = sample["net_input"]["src_tokens"].shape
                src_tokens = sample["net_input"]["src_tokens"].view(len(loss_mask)//option_num, option_num, -1)[:, 0]
                net_output, extra = model(
                    src_tokens,
                    external_qkv=True
                )
                net_output = net_output.unsqueeze(1).expand(-1, option_num, -1, -1).reshape(len(loss_mask), len(src_tokens[0]), -1)
            else: 
                # completion tasks
                net_output, extra = model(
                    sample["net_input"]["src_tokens"],
                    external_qkv=True
                )
                
            net_output = net_output[:, :-1, :]
            net_output = (net_output, extra)
            targets = sample["net_input"]["src_tokens"][:, 1:].unsqueeze(-1)

            assert len(sample["net_input"]["src_tokens"]) // option_num >= 1

            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            loss = torch.gather(lprobs, -1, targets).squeeze(-1) * (loss_mask != False).int()

            loss = loss.sum(-1)
            loss = loss / loss_mask.int().sum(-1)

            fewshot_labels = sample["targets"].view(-1)
            pred_label = torch.argmax(loss.view(-1, option_num), dim=1)
            target_label = fewshot_labels.view(-1, option_num)[:,0]

            logging_output = {}

            logging_output.update(
                {
                    "loss": loss.sum().data,
                    "ntokens": sample["ntokens"],
                    "nsentences": sample_size,
                    "sample_size": sample_size,
                    "ncorrect": (pred_label == target_label).sum(),
                    "npos": (target_label == 0).sum(),
                    "nneg": (target_label == 1).sum(),
                    "npredpos": (pred_label == 0).sum(),
                }
            )
            return loss, sample_size, logging_output
        else:
            # generation tasks
            src_tokens = sample["net_input"]["src_tokens"][0]
            generated_tokens_all = self.generate(sample, beam=self.cfg.beam)

            pred_str = self.decode(generated_tokens_all[0]["tokens"][len(src_tokens)-1:])

            assert self.task.answer_set is not None
            idx = sample['id']
            golden_str_set = self.task.answer_set[idx]
            
            max_f1 = 0
            max_ncorrect = 0
            for golden_str in golden_str_set:
                ncorrect = compute_exact(golden_str, pred_str)
                f1 = compute_f1(golden_str, pred_str)
                if f1 > max_f1:
                    max_f1 = f1
                if ncorrect > max_ncorrect:
                    max_ncorrect = ncorrect
                if max_ncorrect == 1:
                    break

            sample_size = len(sample["net_input"]["src_tokens"])
            logging_output = {
                "loss": 0,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
                "ncorrect": max_ncorrect,
                "f1": max_f1
            }
            
            return 0, sample_size, logging_output
        
    def generate(
        self,
        sample: List[torch.LongTensor],
        beam: int = 3,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:
        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.task.cfg)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.generator

        results = []
        translations = generator.generate(
            self.task.models, sample, prefix_tokens=sample["net_input"]["src_tokens"], bos_token=self.task.dictionary.bos()
        )
        for id, hypos in zip([0], translations):
            results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        return outputs[0]

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            f1 = sum(log.get("f1", 0) for log in logging_outputs)
            total_num = sum(log.get("sample_size", 0) for log in logging_outputs)
            npos = sum(log.get("npos", 0) for log in logging_outputs)
            nneg = sum(log.get("nneg", 0) for log in logging_outputs)
            npredpos = sum(log.get("npredpos", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )
            metrics.log_scalar(
                "f1", 100.0 * f1 / nsentences, nsentences, round=1
            )
            metrics.log_scalar(
                "pos_proportion", 100.0 * npos / nsentences, nsentences, round=1
            )
            metrics.log_scalar(
                "neg_proportion", 100.0 * nneg / nsentences, nsentences, round=1
            )
            metrics.log_scalar(
                "npredpos", 100.0 * npredpos / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True