import random

from typing import Dict, List
from datasets import Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, BatchEncoding

from config import Arguments


def get_prompt_save_path(args: Arguments) -> str:
    from model_utils import parse_model_id
    model_id: str = parse_model_id(args.model_name_or_path)
    out_path: str = '{}/{}_{}_k{}.jsonl.gz'.format(
        args.output_dir, model_id, args.llm_eval_split, args.llm_k_shot
    )

    return out_path


def reward_transform_func(
        tokenizer: PreTrainedTokenizerFast,
        reward_max_length: int,
        corpus: Dataset,
        examples: Dict[str, List]) -> BatchEncoding:
    input_docs: List[str] = []

    # ATTENTION: this code should be consistent with RerankDataLoader
    for doc_id in examples['doc_id']:
        doc_id = int(doc_id)
        input_docs.append(corpus[doc_id]['contents'].strip())

    input_queries = []
    for query, answers, options in zip(examples['query'], examples['answers'], examples['options']):
        current_query = query
        if len(options) > 1:
            current_query += '\n' + options[ord(answers[0]) - ord('A')]
        else:
            current_query += '\n' + random.choice(answers)
        input_queries.append(current_query)

    batch_dict = tokenizer(input_queries,
                           text_pair=input_docs,
                           max_length=reward_max_length,
                           padding=PaddingStrategy.DO_NOT_PAD,
                           return_token_type_ids=False,
                           truncation=True)

    return batch_dict
