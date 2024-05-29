#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
The reader model code + its utilities (loss computation and input batch tensor generator)
"""

import collections
import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor as T
from torch.nn import CrossEntropyLoss

from dpr.data.reader_data import ReaderSample, ReaderPassage
from dpr.utils.model_utils import init_weights

logger = logging.getLogger()

ReaderBatch = collections.namedtuple('ReaderBatch', ['input_ids', 'start_positions', 'end_positions', 'answers_mask'])


class Reader(nn.Module):

    def __init__(self, encoder: nn.Module, hidden_size):
        super(Reader, self).__init__()
        self.encoder = encoder
        self.qa_outputs = nn.Linear(hidden_size, 2)
        self.qa_classifier = nn.Linear(hidden_size, 1)
        init_weights([self.qa_outputs, self.qa_classifier])

    def forward(self, input_ids: T, attention_mask: T, start_positions=None, end_positions=None, answer_mask=None):
        # notations: N - number of questions in a batch, M - number of passages per questions, L - sequence length
        N, M, L = input_ids.size()
        start_logits, end_logits, relevance_logits = self._forward(input_ids.view(N * M, L),
                                                                   attention_mask.view(N * M, L))
        if self.training:
            return compute_loss(start_positions, end_positions, answer_mask, start_logits, end_logits, relevance_logits,
                                N, M)

        return start_logits.view(N, M, L), end_logits.view(N, M, L), relevance_logits.view(N, M)

    def _forward(self, input_ids, attention_mask):
        # TODO: provide segment values
        sequence_output, _pooled_output, _hidden_states = self.encoder(input_ids, None, attention_mask)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        rank_logits = self.qa_classifier(sequence_output[:, 0, :])
        return start_logits, end_logits, rank_logits


def compute_loss(start_positions, end_positions, answer_mask, start_logits, end_logits, relevance_logits, N, M):
    start_positions = start_positions.view(N * M, -1)
    end_positions = end_positions.view(N * M, -1)
    answer_mask = answer_mask.view(N * M, -1)

    start_logits = start_logits.view(N * M, -1)
    end_logits = end_logits.view(N * M, -1)
    relevance_logits = relevance_logits.view(N * M)

    answer_mask = answer_mask.type(torch.FloatTensor).cuda()

    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    loss_fct = CrossEntropyLoss(reduce=False, ignore_index=ignored_index)

    # compute switch loss
    relevance_logits = relevance_logits.view(N, M)
    switch_labels = torch.zeros(N, dtype=torch.long).cuda()
    switch_loss = torch.sum(loss_fct(relevance_logits, switch_labels))

    # compute span loss
    start_losses = [(loss_fct(start_logits, _start_positions) * _span_mask)
                    for (_start_positions, _span_mask)
                    in zip(torch.unbind(start_positions, dim=1), torch.unbind(answer_mask, dim=1))]

    end_losses = [(loss_fct(end_logits, _end_positions) * _span_mask)
                  for (_end_positions, _span_mask)
                  in zip(torch.unbind(end_positions, dim=1), torch.unbind(answer_mask, dim=1))]
    loss_tensor = torch.cat([t.unsqueeze(1) for t in start_losses], dim=1) + \
                  torch.cat([t.unsqueeze(1) for t in end_losses], dim=1)

    loss_tensor = loss_tensor.view(N, M, -1).max(dim=1)[0]
    span_loss = _calc_mml(loss_tensor)
    return span_loss + switch_loss


def create_reader_input(pad_token_id: int,
                        samples: List[ReaderSample],
                        passages_per_question: int,
                        max_length: int,
                        max_n_answers: int,
                        is_train: bool,
                        shuffle: bool,
                        ) -> ReaderBatch:
    """
    Creates a reader batch instance out of a list of ReaderSample-s
    :param pad_token_id: id of the padding token
    :param samples: list of samples to create the batch for
    :param passages_per_question: amount of passages for every question in a batch
    :param max_length: max model input sequence length
    :param max_n_answers: max num of answers per single question
    :param is_train: if the samples are for a train set
    :param shuffle: should passages selection be randomized
    :return: ReaderBatch instance
    """
    input_ids = []
    start_positions = []
    end_positions = []
    answers_masks = []
    empty_sequence = torch.Tensor().new_full((max_length,), pad_token_id, dtype=torch.long)

    for sample in samples:
        positive_ctxs = sample.positive_passages
        negative_ctxs = sample.negative_passages if is_train else sample.passages

        sample_tensors = _create_question_passages_tensors(positive_ctxs,
                                                           negative_ctxs,
                                                           passages_per_question,
                                                           empty_sequence,
                                                           max_n_answers,
                                                           pad_token_id,
                                                           is_train,
                                                           is_random=shuffle)
        if not sample_tensors:
            logger.warning('No valid passages combination for question=%s ', sample.question)
            continue
        sample_input_ids, starts_tensor, ends_tensor, answer_mask = sample_tensors
        input_ids.append(sample_input_ids)
        if is_train:
            start_positions.append(starts_tensor)
            end_positions.append(ends_tensor)
            answers_masks.append(answer_mask)
    input_ids = torch.cat([ids.unsqueeze(0) for ids in input_ids], dim=0)

    if is_train:
        start_positions = torch.stack(start_positions, dim=0)
        end_positions = torch.stack(end_positions, dim=0)
        answers_masks = torch.stack(answers_masks, dim=0)

    return ReaderBatch(input_ids, start_positions, end_positions, answers_masks)


def _calc_mml(loss_tensor):
    marginal_likelihood = torch.sum(torch.exp(
        - loss_tensor - 1e10 * (loss_tensor == 0).float()), 1)
    return -torch.sum(torch.log(marginal_likelihood +
                                torch.ones(loss_tensor.size(0)).cuda() * (marginal_likelihood == 0).float()))


def _pad_to_len(seq: T, pad_id: int, max_len: int):
    s_len = seq.size(0)
    if s_len > max_len:
        return seq[0: max_len]
    return torch.cat([seq, torch.Tensor().new_full((max_len - s_len,), pad_id, dtype=torch.long)], dim=0)


def _get_answer_spans(idx, positives: List[ReaderPassage], max_len: int):
    positive_a_spans = positives[idx].answers_spans
    return [span for span in positive_a_spans if (span[0] < max_len and span[1] < max_len)]


def _get_positive_idx(positives: List[ReaderPassage], max_len: int, is_random: bool):
    # select just one positive
    positive_idx = np.random.choice(len(positives)) if is_random else 0

    if not _get_answer_spans(positive_idx, positives, max_len):
        # question may be too long, find the first positive with at least one valid span
        positive_idx = next((i for i in range(len(positives)) if _get_answer_spans(i, positives, max_len)),
                            None)
    return positive_idx


def _create_question_passages_tensors(positives: List[ReaderPassage], negatives: List[ReaderPassage], total_size: int,
                                      empty_ids: T,
                                      max_n_answers: int,
                                      pad_token_id: int,
                                      is_train: bool,
                                      is_random: bool = True):
    max_len = empty_ids.size(0)
    if is_train:
        # select just one positive
        positive_idx = _get_positive_idx(positives, max_len, is_random)
        if positive_idx is None:
            return None

        positive_a_spans = _get_answer_spans(positive_idx, positives, max_len)[0: max_n_answers]

        answer_starts = [span[0] for span in positive_a_spans]
        answer_ends = [span[1] for span in positive_a_spans]

        assert all(s < max_len for s in answer_starts)
        assert all(e < max_len for e in answer_ends)

        positive_input_ids = _pad_to_len(positives[positive_idx].sequence_ids, pad_token_id, max_len)

        answer_starts_tensor = torch.zeros((total_size, max_n_answers)).long()
        answer_starts_tensor[0, 0:len(answer_starts)] = torch.tensor(answer_starts)

        answer_ends_tensor = torch.zeros((total_size, max_n_answers)).long()
        answer_ends_tensor[0, 0:len(answer_ends)] = torch.tensor(answer_ends)

        answer_mask = torch.zeros((total_size, max_n_answers), dtype=torch.long)
        answer_mask[0, 0:len(answer_starts)] = torch.tensor([1 for _ in range(len(answer_starts))])

        positives_selected = [positive_input_ids]

    else:
        positives_selected = []
        answer_starts_tensor = None
        answer_ends_tensor = None
        answer_mask = None

    positives_num = len(positives_selected)
    negative_idxs = np.random.permutation(range(len(negatives))) if is_random else range(
        len(negatives) - positives_num)

    negative_idxs = negative_idxs[:total_size - positives_num]

    negatives_selected = [_pad_to_len(negatives[i].sequence_ids, pad_token_id, max_len) for i in negative_idxs]

    while len(negatives_selected) < total_size - positives_num:
        negatives_selected.append(empty_ids.clone())

    input_ids = torch.stack([t for t in positives_selected + negatives_selected], dim=0)
    return input_ids, answer_starts_tensor, answer_ends_tensor, answer_mask
