# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# copied partially from https://github.com/yuchenlin/LLM-Blender/blob/main/llm_blender/pair_ranker/pairrm.py
# and added pairwise tokenization function from https://huggingface.co/llm-blender/PairRM-hf
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Model,
    DebertaV2PreTrainedModel,
    SequenceClassifierOutput,
)


def tokenize_conv_pair(tokenizer, convAs: List[str], convBs: List[str], **kwargs):
    """Compare two conversations by takeing USER turns as inputs and ASSISTANT turns as candidates
        Multi-turn conversations comparison is also supportted.
        a conversation format is:
        ```python
        [
            {
                "content": "hello",
                "role": "user"
            },
            {
                "content": "hi",
                "role": "assisstant"
            },
            ...
        ]
        ```
    Args:
        convAs (List[List[dict]]): List of conversations
        convAs (List[List[dict]]): List of conversations
    """

    for c in convAs + convBs:
        if not all([c[i]["role"] == "assistant" for i in range(1, len(c), 2)]):
            print(c)

        assert len(c) % 2 == 0, "Each conversation must have even number of turns"
        assert all([c[i]["role"] == "user" for i in range(0, len(c), 2)]), "Each even turn must be USER"
        assert all([c[i]["role"] == "assistant" for i in range(1, len(c), 2)]), "Each odd turn must be ASSISTANT"
    # check conversations correctness
    assert len(convAs) == len(convBs), "Number of conversations must be the same"
    for c_a, c_b in zip(convAs, convBs):
        assert len(c_a) == len(c_b), "Number of turns in each conversation must be the same"
        assert all(
            [c_a[i]["content"] == c_b[i]["content"] for i in range(0, len(c_a), 2)]
        ), "USER turns must be the same"

    instructions = [
        "Finish the following coversation in each i-th turn by filling in <Response i> with your response."
    ] * len(convAs)
    inputs = [
        "\n".join(["USER: " + x[i]["content"] + f"\nAssistant: <Response {i//2+1}>" for i in range(0, len(x), 2)])
        for x in convAs
    ]
    cand1_texts = [
        "\n".join([f"<Response {i//2+1}>: " + x[i]["content"] for i in range(1, len(x), 2)]) for x in convAs
    ]
    cand2_texts = [
        "\n".join([f"<Response {i//2+1}>: " + x[i]["content"] for i in range(1, len(x), 2)]) for x in convBs
    ]
    inputs = [inst + inp for inst, inp in zip(instructions, inputs)]
    encodings = tokenize_pair(tokenizer, inputs, cand1_texts, cand2_texts, **kwargs)
    return encodings


def tokenize_pair(
    tokenizer,
    sources: List[str],
    candidate1s: List[str],
    candidate2s: List[str],
    source_prefix="<|source|>",
    cand1_prefix="<|candidate1|>",
    cand2_prefix="<|candidate2|>",
    source_max_length=1224,
    candidate_max_length=412,
    **kwargs,
):
    ids = []
    assert len(sources) == len(candidate1s) == len(candidate2s)
    max_length = source_max_length + 2 * candidate_max_length
    for i in range(len(sources)):
        source_ids = tokenizer.encode(source_prefix + sources[i], max_length=source_max_length, truncation=True)
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = tokenizer.encode(
            cand1_prefix + candidate1s[i], max_length=candidate_max_length, truncation=True
        )
        candidate2_ids = tokenizer.encode(
            cand2_prefix + candidate2s[i], max_length=candidate_max_length, truncation=True
        )
        ids.append(source_ids + candidate1_ids + candidate2_ids)
    encodings = tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
    return encodings


class PairRMPipeline:
    """
    This class outputs a delta rather than a score for each.
    """

    def __init__(self, task, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # turn off gradients for model and set in eval mode
        self.model.eval().requires_grad_(False)

    def __call__(self, candidates_A: List[str], candidates_B: List[str], output_logits=False, **kwargs):
        AB_encodings = tokenize_conv_pair(self.tokenizer, candidates_A, candidates_B, **kwargs)
        AB_outputs = self.model(**AB_encodings.to(self.model.device))
        AB_logits = AB_outputs.logits
        BA_encodings = tokenize_conv_pair(self.tokenizer, candidates_B, candidates_A, **kwargs)
        BA_outputs = self.model(**BA_encodings.to(self.model.device))
        BA_logits = BA_outputs.logits
        logits = AB_logits - BA_logits
        if output_logits:
            return logits.tolist()
        else:
            return logits > 0


class DebertaV2PairRM(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out

        # LM
        self.pretrained_model = DebertaV2Model(config)
        self.hidden_size = config.hidden_size

        self.sep_token_id = config.sep_token_id  # to add
        self.source_prefix_id = config.source_prefix_id  # to add
        self.cand_prefix_id = config.cand_prefix_id
        self.cand1_prefix_id = config.cand1_prefix_id
        self.cand2_prefix_id = config.cand2_prefix_id

        self.head_layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2 * self.hidden_size, 1 * self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.hidden_size, self.n_tasks),
        )
        self.sigmoid = nn.Sigmoid()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #  <source_prefix_id>...<sep><cand1_prefix_id>...<sep><cand2_prefix_id> ... <sep>
        assert all(
            [self.source_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
        ), "<source> id not in input_ids"
        assert all(
            [self.cand1_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
        ), "<candidate1> id not in input_ids"
        assert all(
            [self.cand2_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
        ), "<candidate2> id not in input_ids"

        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand1_idxs = torch.where(input_ids == self.cand1_prefix_id)
        cand1_encs = encs[cand1_idxs[0], cand1_idxs[1], :]
        cand2_idxs = torch.where(input_ids == self.cand2_prefix_id)
        cand2_encs = encs[cand2_idxs[0], cand2_idxs[1], :]

        # reduce
        source_cand1_encs = torch.cat([source_encs, cand1_encs], dim=-1)
        source_cand2_encs = torch.cat([source_encs, cand2_encs], dim=-1)
        left_pred_scores = self.head_layer(source_cand1_encs)
        right_pred_scores = self.head_layer(source_cand2_encs)

        loss = None
        if labels is not None:
            loss = self.compute_loss(left_pred_scores, right_pred_scores, labels)

        preds = (left_pred_scores - right_pred_scores).mean(dim=-1)
        return SequenceClassifierOutput(
            loss=loss,
            logits=preds,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )

    def compute_loss(self, left_pred_scores, right_pred_scores, labels):
        """
        Args:
            left_pred_scores: [n_candidates, n_task]
            right_pred_scores: [n_candidates, n_task]
            labels: [n_candidates, n_task], 1/0/-1 for left/right/both is better
        """

        device = left_pred_scores.device
        loss = torch.tensor(0.0).to(left_pred_scores.device)

        dif_scores = labels
        left_pred_scores = left_pred_scores * dif_scores.sign()
        right_pred_scores = -right_pred_scores * dif_scores.sign()
        cls_loss = torch.tensor(0.0, device=device)
        cls_loss += -torch.log(torch.sigmoid(left_pred_scores + right_pred_scores)).mean()
        loss += cls_loss
        return loss
