import random

from typing import List, Dict, Optional
from datasets import load_dataset, Dataset
from openai.types.chat import ChatCompletion
from contextlib import nullcontext
from transformers import PreTrainedTokenizerFast
from threading import Lock

from config import Arguments
from utils import batch_truncate
from logger_config import logger


def load_corpus() -> Dataset:
    corpus: Dataset = load_dataset('corag/kilt-corpus', split='train')
    logger.info(f'Loaded {len(corpus)} passages from corag/kilt-corpus')
    return corpus


def log_random_samples(dataset: Dataset, num_samples: int = 3):
    from utils import log_truncate
    # Log a few random samples
    num_samples = min(num_samples, len(dataset))
    for index in random.sample(range(len(dataset)), num_samples):
        logger.info(f"\nSample {index} of the dataset:")
        for key, value in dataset[index].items():
            logger.info(f"################ {key}")
            logger.info(log_truncate(value))


def format_input_context(doc: Dict[str, str]) -> str:
    title: str = doc.get('title', '')
    contents: str = doc['contents']
    if contents.startswith(title + '\n'):
        contents = contents[len(title) + 1:]

    return f'{title}\n{contents}'.strip()


def parse_answer_logprobs(response: ChatCompletion) -> List[float]:
    prompt_logprobs: List[Dict] = response.prompt_logprobs[::-1]

    # Hacky: this only works for llama-3 models
    assert '128006' in prompt_logprobs[3], f"Invalid prompt logprobs: {prompt_logprobs}"
    prompt_logprobs = prompt_logprobs[4:] # Skip the added generation prompt

    answer_logprobs: List[float] = []
    for logprobs in prompt_logprobs:
        logprobs: Dict[str, Dict]
        prob_infos: List[Dict] = sorted(list(logprobs.values()), key=lambda x: x['rank'])
        if prob_infos[-1]['decoded_token'] == '<|end_header_id|>':
            # also skip the "\n\n" token
            answer_logprobs = answer_logprobs[:-1]
            break

        prob = prob_infos[-1]['logprob']
        answer_logprobs.append(prob)

    return answer_logprobs


def _apply_context_placement_strategy(context_placement: str, contexts: List[str]) -> List[str]:
    # Assume the contexts are sorted by retrieval scores descending
    if context_placement == 'forward':
        return contexts
    elif context_placement == 'backward':
        return list(reversed(contexts))
    elif context_placement == 'random':
        random.shuffle(contexts)
        return contexts
    else:
        raise ValueError(f'Invalid context placement strategy: {context_placement}')


def format_documents_for_final_answer(
        args: Arguments,
        context_doc_ids: List[str],
        tokenizer: PreTrainedTokenizerFast, corpus: Dataset,
        lock: Optional[Lock] = None
) -> List[str]:
    selected_doc_ids: List[str] = context_doc_ids[:args.num_contexts]
    documents: List[str] = [format_input_context(corpus[int(doc_id)]) for doc_id in selected_doc_ids]

    max_per_ctx_length: int = int(args.max_len / max(args.num_contexts, 1) * 1.2)
    with nullcontext() if lock is None else lock:
        documents = batch_truncate(documents, tokenizer=tokenizer, max_length=max_per_ctx_length)
    documents = _apply_context_placement_strategy(args.context_placement, documents)

    return documents
