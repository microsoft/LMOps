import os
import json
import numpy as np

from typing import Dict, List, Optional
from transformers import AutoTokenizer
from datasets import Dataset

from config import Arguments
from logger_config import logger
from tasks import parse_decoded_text_by_task, get_metric_name_by_task_name, get_possible_answers_by_task_name
from evaluation.metrics import compute_metrics
from llms import BaseLLM
from model_utils import parse_model_id
from data_utils import save_llm_decoding_results
from utils import save_json_to_file, DictTrie, build_trie, wait_until_all_files_show_up


class LLMEvaluator:
    def __init__(self, args: Arguments, llm: BaseLLM):
        self.args = args
        self.llm = llm
        self.model_id: str = parse_model_id(self.args.model_name_or_path)
        self.llm_model_id: str = parse_model_id(self.args.llm_model_name_or_path)

    def eval_single_task(self, eval_dataset: Dataset, task_name: str):
        out_path: str = '{}/{}/{}_{}_metrics.json'.format(self.args.output_dir, self.llm_model_id, task_name, self.model_id)
        if os.path.exists(out_path):
            logger.info('Task {} has already been evaluated'.format(task_name))
            return

        task_ds: Dataset = eval_dataset.filter(lambda x: x['task_name'] == task_name)
        logger.info('Task: {}, # of examples: {}'.format(task_name, len(task_ds)))
        if len(task_ds) == 0:
            logger.error('No examples for task: {}'.format(task_name))
            return
        sharded_task_ds = task_ds.shard(num_shards=self.args.world_size, index=self.args.process_index, contiguous=True)
        logger.info('Worker {} needs to process {} examples'.format(self.args.process_index, len(task_ds)))

        queries: List[str] = sharded_task_ds['query']
        input_prompts: List[str] = sharded_task_ds['input_prompt']
        options_list: List[List[str]] = sharded_task_ds['options']
        assert len(input_prompts) == len(queries)
        assert all(not q.endswith('\n') for q in queries)
        assert all(len(options) == len(options_list[0]) for options in options_list)

        # prompt may be empty in the zero-shot setting
        input_texts: List[str] = [
            '{}\n\n{}\n'.format(prompt, query) if prompt else '{}\n'.format(query) for prompt, query in
            zip(input_prompts, queries)
        ]
        possible_answers: Optional[List[str]] = get_possible_answers_by_task_name(task_name)

        if len(options_list[0]) <= 1:
            # classification or open-ended generation tasks
            prefix_trie: Optional[DictTrie] = None
            if possible_answers and self.args.llm_constrained_decoding:
                tokenizer = AutoTokenizer.from_pretrained(self.args.llm_model_name_or_path)
                possible_answers = ['{}\n'.format(ans) for ans in possible_answers]
                prefix_trie: DictTrie = build_trie(tokenizer=tokenizer, output_texts=possible_answers)
                logger.info('Task: {}, constrained generation targets: {}'.format(task_name, possible_answers))

            decoded_texts: List[str] = self.llm.batch_decode(input_texts, prefix_trie=prefix_trie)
        else:
            # multiple-choice tasks
            assert len(options_list[0]) == len(possible_answers)
            choices: List[str] = sum(options_list, [])
            scoring_inputs = sum(
                [[input_text.strip() for _ in range(len(possible_answers))] for input_text in input_texts], [])
            scores: List[float] = self.llm.batch_score(scoring_inputs, choices, delimiter='\n')
            answer_indices = np.argmax(np.array(scores).reshape(-1, len(possible_answers)), axis=1)
            decoded_texts: List[str] = [possible_answers[idx] for idx in answer_indices]

        parsed_decoded_texts: List[str] = [
            parse_decoded_text_by_task(decoded_text, task_name) for decoded_text in decoded_texts
        ]
        save_json_to_file({
            'input_texts': input_texts,
            'decoded_texts': decoded_texts,
            'parsed_decoded_texts': parsed_decoded_texts,
        }, self._get_tmp_path(self.args.process_index, task_name))

        if self.args.process_index <= 0:
            wait_until_all_files_show_up(
                [self._get_tmp_path(worker_idx, task_name) for worker_idx in range(self.args.world_size)]
            )
            self._compute_metrics(task_ds=task_ds, task_name=task_name, out_path=out_path)

    def _compute_metrics(self, task_ds: Dataset, task_name: str, out_path: str):
        # merge results from all workers
        input_texts: List[str] = []
        decoded_texts: List[str] = []
        parsed_decoded_texts: List[str] = []
        for worker_idx in range(self.args.world_size):
            tmp_path: str = self._get_tmp_path(worker_idx, task_name)
            tmp_results: Dict = json.load(open(tmp_path, 'r', encoding='utf-8'))
            input_texts.extend(tmp_results['input_texts'])
            decoded_texts.extend(tmp_results['decoded_texts'])
            parsed_decoded_texts.extend(tmp_results['parsed_decoded_texts'])

        answers: List = task_ds['answers']
        if max(len(answer) for answer in answers) == 1:
            # single answer
            answers: List[str] = [answer[0] for answer in answers]

        metric_name: str = get_metric_name_by_task_name(task_name)
        metrics: Dict = compute_metrics(metric=metric_name, labels=answers, preds=parsed_decoded_texts)
        metrics.update({
            'model_name_or_path': self.args.model_name_or_path,
            'llm_model_name_or_path': self.args.llm_model_name_or_path,
            'n_shot': self.args.llm_k_shot,
            'task_name': task_name,
        })
        logger.info('Task {}, metric {}: {}'.format(task_name, metric_name, json.dumps(metrics)))

        save_json_to_file(metrics, out_path)
        model_id: str = parse_model_id(self.args.model_name_or_path)
        llm_model_id: str = parse_model_id(self.args.llm_model_name_or_path)
        save_llm_decoding_results(
            out_path='{}/{}/{}_{}_decoding_results.jsonl'.format(self.args.output_dir, llm_model_id, task_name, model_id),
            input_texts=input_texts,
            decoded_texts=decoded_texts,
            parsed_decoded_texts=parsed_decoded_texts,
            options_list=task_ds['options'],
            answer_texts=answers,
        )

        for worker_idx in range(self.args.world_size):
            tmp_path: str = self._get_tmp_path(worker_idx, task_name)
            os.remove(tmp_path)

    def _get_tmp_path(self, worker_idx: int, task_name: str) -> str:
        tmp_dir = self.args.output_dir if self.args.world_size <= 1 else 'tmp/'
        llm_model_id: str = parse_model_id(self.args.llm_model_name_or_path)
        return '{}/{}/{}_{}.json'.format(tmp_dir, llm_model_id, task_name, worker_idx)
