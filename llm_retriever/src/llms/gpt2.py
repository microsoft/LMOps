import torch
import tqdm
import numpy as np

from contextlib import nullcontext
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from typing import List
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2TokenizerFast
from transformers.generation.utils import GreedySearchDecoderOnlyOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from utils import move_to_device
from logger_config import logger
from config import Arguments
from llms.base_llm import BaseLLM
from collators.gpt2_collator import ScoreCollator, DecodeCollator


class GPT2(BaseLLM):

    def __init__(self, args: Arguments, model_name_or_path: str = 'gpt2-xl', **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        self.args = args
        self.tokenizer: GPT2TokenizerFast = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.truncation_side = 'left'
        self.batch_size_per_device = args.llm_batch_size_per_device

        dtype = torch.float16 if args.fp16 else torch.float32
        self.model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.eval()

    @torch.no_grad()
    def batch_score(
            self, input_texts: List[str], output_texts: List[str],
            delimiter: str = '\n', **kwargs
    ) -> List[float]:
        assert len(input_texts) == len(output_texts), '{} != {}'.format(len(input_texts), len(output_texts))
        assert not all(output in ['A', 'B', 'C', 'D'] for output in output_texts), 'output_texts should not be letters'

        collator = ScoreCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.llm_max_input_length,
            pad_to_multiple_of=8,
            delimiter=delimiter,
        )

        dataset = Dataset.from_dict({
            'input_texts': input_texts,
            'output_texts': output_texts
        })
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=2,
            collate_fn=collator,
            pin_memory=True
        )

        avg_log_probs: List[float] = []
        for batch_dict in tqdm.tqdm(data_loader, desc='batch score', mininterval=10, disable=len(dataset) < 1024):
            # Hack: remove token_type_ids for llama model
            if 'llama' in self.model_name_or_path and 'token_type_ids' in batch_dict:
                del batch_dict['token_type_ids']

            batch_dict = move_to_device(batch_dict, device=self.model.device)
            with torch.cuda.amp.autocast() if self.args.fp16 else nullcontext():
                outputs: CausalLMOutputWithCrossAttentions = self.model(
                    **batch_dict, return_dict=True, use_cache=False
                )

                labels = batch_dict['labels']
                # Shift so that tokens < n predict n
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction='none')
                per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                per_sequence_loss = per_token_loss.view(batch_dict['input_ids'].size(0), -1).sum(dim=1)
                # divide by the number of valid labels
                num_valid_labels = torch.sum(labels != -100, dim=1).float()
                avg_log_probs += (-per_sequence_loss / num_valid_labels).cpu().tolist()

                logger.debug('num_valid_labels: {}, loss: {}, per_token_loss: {}, avg_per_token_loss: {}'.format(
                    num_valid_labels, outputs.loss, per_token_loss,
                    per_token_loss.sum() / torch.sum(labels != -100).float())
                )

        return avg_log_probs

    def batch_decode(self, input_texts: List[str], prefix_trie=None, **kwargs) -> List[str]:
        collator = DecodeCollator(
            tokenizer=self.tokenizer,
            max_length=self.args.llm_max_input_length,
            pad_to_multiple_of=8
        )
        dataset: Dataset = Dataset.from_dict({'input_texts': input_texts})
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            num_workers=2,
            collate_fn=collator,
            pin_memory=True
        )

        decoded_texts: List[str] = []
        eos_token_id: int = self.tokenizer.encode('\n')[-1]
        for batch_dict in tqdm.tqdm(data_loader, mininterval=10, desc='batch decode'):
            # Hack: remove token_type_ids for llama model
            if 'llama' in self.model_name_or_path and 'token_type_ids' in batch_dict:
                del batch_dict['token_type_ids']

            batch_dict = move_to_device(batch_dict, device=self.model.device)
            input_len: int = batch_dict['input_ids'].shape[1]

            def _prefix_allowed_tokens_fn(_, generated_ids):
                return prefix_trie.get(generated_ids.tolist()[input_len:])

            with torch.cuda.amp.autocast() if self.args.fp16 else nullcontext():
                outputs: GreedySearchDecoderOnlyOutput = self.model.generate(
                    **batch_dict,
                    num_beams=1,
                    do_sample=False,
                    max_new_tokens=self.args.llm_max_decode_length,
                    begin_suppress_tokens=[eos_token_id],
                    eos_token_id=eos_token_id,
                    prefix_allowed_tokens_fn=_prefix_allowed_tokens_fn if prefix_trie else None,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                generated_token_ids = outputs.sequences[:, input_len:]
                logger.debug('generated_token_ids: {}'.format(generated_token_ids.tolist()))

                if outputs.scores is not None:
                    transition_scores = self.model.compute_transition_scores(
                        outputs.sequences, outputs.scores, normalize_logits=True
                    )
                    for tok, score in zip(generated_token_ids[0].cpu(), transition_scores[0].cpu()):
                        if tok in self.tokenizer.all_special_ids:
                            continue
                        # | token | token string | logits | probability
                        logger.info(f"| {tok:5d} | {self.tokenizer.decode(tok):8s} | {score.numpy():.4f} "
                                    f"| {np.exp(score.numpy()):.2%}")

            decoded_texts += self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)

        return decoded_texts
