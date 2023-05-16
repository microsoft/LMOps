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
import random
import torch.nn.functional as F

from typing import Any, Dict, Iterator, List
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from struprompting.data.squad.squad_metrics import compute_f1, compute_exact


@dataclass
class FewshotEvalConfig(FairseqDataclass):
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


@register_criterion("fs_eval", dataclass=FewshotEvalConfig)
class FewshotEvalCriterion(FairseqCriterion):
    def __init__(self, cfg: FewshotEvalConfig, task):
        super().__init__(task)
        self.cfg = cfg
        self.is_generation = cfg.is_generation
        # acc debug
        self.valid_num_sum = 0
        self.acc_record = 0

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
        option_num = self.task.fewshot_task.class_num
        assert sample["targets"].size(0) % option_num == 0
        sample_size = sample["targets"].size(0) // option_num
        loss_mask = sample["net_input"]["gpt_loss_mask"][:, 1:]
        context_len = sample["context_len"]

        if len(loss_mask) == loss_mask.int().sum():
            # single-token classification tasks
            shape = sample["net_input"]["src_tokens"].shape
            src_tokens = sample["net_input"]["src_tokens"].view(len(loss_mask)//option_num, option_num, -1)[:, 0]
            net_output, extra = model(
                src_tokens
            )
            net_output = net_output.unsqueeze(1).expand(-1, option_num, -1, -1).reshape(len(loss_mask), len(src_tokens[0]), -1)
        else: 
            # multi-token classification tasks and multi-choice tasks
            net_output, extra = model(
                sample["net_input"]["src_tokens"]
            )
            
        net_output = net_output[:, :-1, :]
        net_output = (net_output, extra)
        targets = sample["net_input"]["src_tokens"][:, 1:].unsqueeze(-1)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        loss = torch.gather(lprobs, -1, targets).squeeze(-1) * (loss_mask != False).int()
        loss = loss.sum(-1)
        loss = loss / loss_mask.int().sum(-1)  # (bsz, option_num)

        fewshot_labels = sample["targets"].view(-1)
        pred_label = torch.argmax(loss.view(-1, option_num), dim=1)
        target_label = fewshot_labels.view(-1, option_num)[:, 0]

        loss = loss.view(-1, option_num)
        tmp_loss = 0
        for i in range(sample_size):
            tmp_loss += loss[i][target_label[i]]
        loss = tmp_loss

        logging_output = {}

        logging_output.update(
            {
                "loss": -loss.sum().data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
                "ncorrect": (pred_label == target_label).sum(),
                "npos": (target_label == 0).sum(),
                "nneg": (target_label == 1).sum(),
            }
        )
        
        # ! ICL analysis
        def process_hiddens_classification(hiddens):
            # single-token classification tasks
            hiddens = torch.stack(hiddens, dim=0)  # (n_layers, len, bsz, hidden)
            hiddens = hiddens.transpose(1, 2)  # (n_layers, bsz, len, hidden)
            record_hiddens = hiddens[:, 0, -1, :]  # (n_layers, hidden)
            return record_hiddens


        def process_attention_map_classification(attn_map, c_len):
            # single-token classification tasks
            attn_map = torch.stack(attn_map, dim=0)  # (n_layers, n_heads, len, len)
            record_attn_map = attn_map[:, :, -1, c_len:]  # (n_layers, n_heads, query_len)
            record_attn_map_ctx = attn_map[:, :, -1, :c_len]  # (n_layers, n_heads, context_len)
            # record_attn_map = record_attn_map.sum(dim=-2)  # (n_layers, query_len)
            record_attn_map_ctx = record_attn_map_ctx.sum(dim=-2)  # (n_layers, context_len)
            return record_attn_map, record_attn_map_ctx


        def process_attention_q_classification(attn_q):
            # single-token classification tasks
            attn_q = torch.stack(attn_q, dim=0)  # (n_layers, n_token, bzs (1), q_hidden_dim)
            record_attn_q = attn_q[:, :, -1, :]  # (n_layers, n_token, q_hidden_dim)
            return record_attn_q


        record_self_attn_out_hiddens = None
        record_trm_out_hiddens = None
        record_attn_map = None
        record_attn_map_ctx = None
        attn_q = None
        
        record_info = None
        
        if len(loss_mask) == loss_mask.int().sum():
            # single-token classification tasks
            record_self_attn_out_hiddens = process_hiddens_classification(extra['self_attn_out_hiddens'])
            qkv_val = extra['qkv_val']
            attn_map = [item['record_attn_map'] for item in qkv_val]
            record_attn_map, record_attn_map_ctx = process_attention_map_classification(attn_map, context_len[0])
            attn_q = [item['q'] for item in qkv_val]
            record_attn_q = process_attention_q_classification(attn_q)
        
        record_info = {
            'self_attn_out_hiddens': record_self_attn_out_hiddens.tolist(),
            'attn_map': record_attn_map.tolist(),
            'attn_map_ctx': record_attn_map_ctx.tolist(),
            'attn_q': record_attn_q,
            'gold_label': target_label[0].item(),
            'pred_label': pred_label[0].item(),
        }
        
        return loss, sample_size, logging_output, record_info
        
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
            npos = sum(log.get("npos", 0) for log in logging_outputs)
            nneg = sum(log.get("nneg", 0) for log in logging_outputs)
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

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True