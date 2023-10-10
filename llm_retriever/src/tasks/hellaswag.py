import re

from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map, to_letter
from tasks.base_task import BaseTask


@task_map.add("hellaswag")
class Hellaswag(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('hellaswag', split=split)

        def _map_func(ex: Dict) -> Dict:
            ex['ctx'] = re.sub(r'\[.*?\]\s', '', ex['ctx']).strip()
            ex['options'] = [re.sub(r'\[.*?\]\s', '', option) for option in ex['endings']]
            return ex

        dataset = dataset.map(_map_func)
        dataset = dataset.rename_column('ctx', 'context')

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("What happens next in this paragraph? {context}", "{answer}"),
            ("Continue writing the next sentence in this paragraph: {context}", "{answer}"),
            ("Continue writing the next sentence. {context}", "{answer}"),
            ("This is a test of commonsense. Complete the next sentence: {context}", "{answer}"),
            ("Write the next sentence in this paragraph: {context}", "{answer}"),
            ("How does the next paragraph end? {context}", "{answer}"),
            ("What most naturally follows? {context}", "{answer}"),
            ("What happens next? {context}", "{answer}"),
            ("What is the most logical next event? {context}", "{answer}"),
            ("Write the next sentence in the following story. {context}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['A', 'B', 'C', 'D']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'hellaswag'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return to_letter(example['label'])
