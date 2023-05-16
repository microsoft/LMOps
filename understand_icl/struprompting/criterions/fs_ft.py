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
class FewshotFTConfig(FairseqDataclass):
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


@register_criterion("fs_ft", dataclass=FewshotFTConfig)
class FewshotFTCriterion(FairseqCriterion):
    def __init__(self, cfg: FewshotFTConfig, task):
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
        loss_mask = sample["net_input"]["gpt_loss_mask"][:, 1:]
        sample_size = sample["net_input"]["src_tokens"].size(0)
        net_output, extra = model(
            sample["net_input"]["src_tokens"]
        )
        net_output = net_output[:, :-1, :]
        net_output = (net_output, extra)
        targets = sample["net_input"]["src_tokens"][:, 1:].unsqueeze(-1)

        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        loss = torch.gather(lprobs, -1, targets).squeeze(-1) * (loss_mask != False).int()
        loss = -loss.sum()

        optim_size = loss_mask.int().sum()

        logging_output = {}

        logging_output.update(
            {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": optim_size,
            }
        )


        # ! ICL analysis
        def process_attention_q_classification(attn_q):
            # single-token classification tasks
            attn_q = torch.stack(attn_q, dim=0)  # (n_layers, n_token, bzs (1), q_hidden_dim)
            record_attn_q = attn_q[:, :, -1, :]  # (n_layers, n_token, q_hidden_dim)
            return record_attn_q

        record_attn_q = None

        record_info = None

        qkv_val = extra['qkv_val']
        attn_q = [item['q'] for item in qkv_val]
        record_attn_q = process_attention_q_classification(attn_q)

        record_info = {
            'attn_q': record_attn_q,
        }

        return loss, optim_size, logging_output, record_info
        
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
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )
        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True