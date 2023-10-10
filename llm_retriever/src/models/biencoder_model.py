import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from torch import Tensor
from transformers import PreTrainedModel, AutoModel
from transformers.modeling_outputs import ModelOutput

from config import Arguments
from logger_config import logger
from utils import dist_gather_tensor, select_grouped_indices, full_contrastive_scores_and_labels, pool


@dataclass
class BiencoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiencoderModel(nn.Module):
    def __init__(self, args: Arguments,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.args = args

        from trainers import BiencoderTrainer
        self.trainer: Optional[BiencoderTrainer] = None

        self._freeze_position_embedding_if_needed(self.lm_q)
        self._freeze_position_embedding_if_needed(self.lm_p)

    def forward(self, batch_dict: Dict[str, Tensor]) -> BiencoderOutput:
        assert self.args.process_index >= 0

        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(batch_dict)

        start = self.args.process_index * q_reps.shape[0]
        group_indices = select_grouped_indices(scores=scores,
                                               group_size=self.args.train_n_passages,
                                               start=start * self.args.train_n_passages)

        if not self.args.do_kd_biencoder:
            # training biencoder from scratch
            loss = self.cross_entropy(scores, labels)
        else:
            # training biencoder with kd
            # batch_size x train_n_passage
            group_scores = torch.gather(input=scores, dim=1, index=group_indices)
            assert group_scores.shape[1] == self.args.train_n_passages
            group_log_scores = torch.log_softmax(group_scores, dim=-1)
            kd_log_target = torch.log_softmax(batch_dict['kd_labels'], dim=-1)

            kd_loss = self.kl_loss_fn(input=group_log_scores, target=kd_log_target)
            ce_loss = self.cross_entropy(scores, labels)
            loss = self.args.kd_cont_loss_weight * ce_loss + kd_loss

        total_n_psg = self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                               labels=labels.contiguous(),
                               scores=scores[:, :total_n_psg].contiguous())

    def _compute_scores(self, batch_dict: Dict[str, Tensor]) -> Tuple:
        embeds = self._encode(self.lm_p, batch_dict)
        batch_size = batch_dict['input_ids'].shape[0] // (self.args.train_n_passages + 1)
        q_reps = embeds[:batch_size]
        p_reps = embeds[batch_size:]
        assert p_reps.shape[0] == q_reps.shape[0] * self.args.train_n_passages

        all_q_reps = dist_gather_tensor(q_reps)
        all_p_reps = dist_gather_tensor(p_reps)
        assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        all_scores, all_labels = full_contrastive_scores_and_labels(
            query=all_q_reps, key=all_p_reps,
            use_all_pairs=self.args.full_contrastive_loss
        )
        if self.args.l2_normalize:
            all_scores = all_scores / self.args.t

        start = self.args.process_index * q_reps.shape[0]
        local_query_indices = torch.arange(start, start + q_reps.shape[0], dtype=torch.long).to(q_reps.device)
        # batch_size x (world_size x batch_size x train_n_passage)
        scores = all_scores.index_select(dim=0, index=local_query_indices)
        labels = all_labels.index_select(dim=0, index=local_query_indices)

        return scores, labels, q_reps, p_reps, all_scores, all_labels

    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['labels', 'kd_labels']}, return_dict=True)
        embeds = pool(last_hidden_states=outputs.last_hidden_state,
                      attention_mask=input_dict['attention_mask'],
                      pool_type=self.args.pool_type)
        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds.contiguous()

    def _freeze_position_embedding_if_needed(self, model: nn.Module):
        if self.args.freeze_position_embedding:
            for name, param in model.named_parameters():
                if 'position_embeddings' in name:
                    param.requires_grad = False
                    logger.info('Freeze {}'.format(name))

    def gradient_checkpointing_enable(self):
        self.lm_q.gradient_checkpointing_enable()

    @classmethod
    def build(cls, args: Arguments, **hf_kwargs):
        if os.path.isdir(args.model_name_or_path):
            logger.info(f'loading shared model weight from {args.model_name_or_path}')
        lm_q = AutoModel.from_pretrained(args.model_name_or_path)
        lm_p = lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)
        return model

    def save(self, output_dir: str):
        self.lm_q.save_pretrained(output_dir)
