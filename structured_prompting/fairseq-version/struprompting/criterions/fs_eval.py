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
                    src_tokens
                )
                net_output = net_output.unsqueeze(1).expand(-1, option_num, -1, -1).reshape(len(loss_mask), len(src_tokens[0]), -1)
            else: 
                # multi-choice tasks
                net_output, extra = model(
                    sample["net_input"]["src_tokens"]
                )
                
            net_output = net_output[:, :-1, :]
            net_output = (net_output, extra)
            targets = sample["net_input"]["src_tokens"][:, 1:].unsqueeze(-1)

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