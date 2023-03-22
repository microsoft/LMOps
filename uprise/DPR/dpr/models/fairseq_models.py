#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on Fairseq code
"""

import logging
from typing import Tuple

from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaModel as FaiseqRobertaModel
from fairseq.optim.adam import FairseqAdam
from torch import Tensor as T
from torch import nn

from dpr.models.hf_models import get_roberta_tensorizer
from .biencoder import BiEncoder

logger = logging.getLogger(__name__)


def get_roberta_biencoder_components(args, inference_only: bool = False, **kwargs):
    question_encoder = RobertaEncoder.from_pretrained(args.pretrained_file)
    ctx_encoder = RobertaEncoder.from_pretrained(args.pretrained_file)
    biencoder = BiEncoder(question_encoder, ctx_encoder)
    optimizer = get_fairseq_adamw_optimizer(biencoder, args) if not inference_only else None

    tensorizer = get_roberta_tensorizer(args)

    return tensorizer, biencoder, optimizer


def get_fairseq_adamw_optimizer(model: nn.Module, args):
    setattr(args, 'lr', [args.learning_rate])
    return FairseqAdam(args, model.parameters()).optimizer


class RobertaEncoder(nn.Module):

    def __init__(self, fairseq_roberta_hub: RobertaHubInterface):
        super(RobertaEncoder, self).__init__()
        self.fairseq_roberta = fairseq_roberta_hub

    @classmethod
    def from_pretrained(cls, pretrained_dir_path: str):
        model = FaiseqRobertaModel.from_pretrained(pretrained_dir_path)
        return cls(model)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        roberta_out = self.fairseq_roberta.extract_features(input_ids)
        cls_out = roberta_out[:, 0, :]
        return roberta_out, cls_out, None

    def get_out_size(self):
        raise NotImplementedError
