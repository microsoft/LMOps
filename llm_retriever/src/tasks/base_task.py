import re

from typing import List, Optional, Tuple, Dict, Union
from datasets import Dataset

from logger_config import logger


class BaseTask(object):

    def __init__(self, template_idx: int = 0, **kwargs):
        self.template_idx = template_idx

    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        raise NotImplementedError

    def get_task_data(self, split: str) -> Optional[Dataset]:
        # columns: query_id / query / answers / task_name
        dataset = self._load_raw_data(split)
        if not dataset:
            return None

        logger.info('Load dataset: {}, split: {}'.format(self.task_name, split))
        dataset = dataset.map(self.map_single, num_proc=4)
        dataset = dataset.add_column(
            'query_id', ['{}_{}_{}'.format(self.task_name, split, idx) for idx in range(len(dataset))]
        )
        dataset = dataset.remove_columns(
            column_names=[col for col in dataset.column_names
                          if col not in ['query_id', 'query', 'answers', 'options', 'task_name']]
        )

        return dataset

    def get_corpus(self) -> Optional[Dataset]:
        # columns: contents / task_name
        corpus = self.get_task_data(split='train')
        if not corpus:
            return None

        def _map_func(example: Dict) -> Dict:
            answer = example['answers'][0]
            if len(example['options']) > 1:
                # multiple-choice tasks
                option_idx = self.possible_answers.index(answer)
                answer = example['options'][option_idx]
            return {
                'contents': '\n'.join([example['query'], answer]),
            }

        corpus = corpus.map(
            _map_func, num_proc=4,
            remove_columns=[col for col in corpus.column_names if col not in ['contents', 'task_name']],
            desc='{} corpus'.format(self.task_name)
        )

        return corpus

    def get_template(self) -> Tuple[str, str]:
        return self.templates[self.template_idx % len(self.templates)]

    def map_single(self, example: Dict) -> Dict:
        # ("If \"{premise}\", can we conclude that \"{hypothesis}\"? Yes, No, or Maybe?", "{answer}"),
        query_template, answer_template = self.get_template()

        # find the key in {key} format using regular expression
        query_keys = re.findall(r'\{(\w+)\}', query_template)
        answer_keys = re.findall(r'\{(\w+)\}', answer_template)

        # replace the key with the value in the example
        query: str = query_template.format(**{key: example[key] for key in query_keys})
        assert len(answer_keys) == 1, "Only one answer key is allowed"
        answer_key = answer_keys[0]
        del answer_keys

        example[answer_key]: Union[str, List[str]] = self.get_answer(example)
        if isinstance(example[answer_key], str):
            answers: List[str] = [answer_template.format(**{answer_key: example[answer_key]})]
        elif isinstance(example[answer_key], list):
            answers: List[str] = [answer_template.format(**{answer_key: ans}) for ans in example[answer_key]]
        else:
            raise ValueError(f"Unknown answer type: {example[answer_key]}")

        return {
            'query': query,
            'options': example.get('options', ['']),
            'answers': answers,
            'task_name': self.task_name,
        }

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        # Many tasks need to override this default implementation
        assert int(example['label']) >= 0, "label must be non-negative"
        return self.possible_answers[int(example['label'])]

    @property
    def templates(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    @property
    def possible_answers(self) -> Optional[List[str]]:
        raise NotImplementedError

    @property
    def metric_name(self) -> str:
        raise NotImplementedError

    @property
    def task_name(self) -> str:
        raise NotImplementedError
