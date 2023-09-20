import re

from typing import Optional, List, Tuple, Dict
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("yelp")
class Yelp(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('yelp_polarity', split=split)

        def _map_func(ex: Dict) -> Dict:
            ex['text'] = re.sub(r'\\\"', '', ex['text'])
            ex['text'] = re.sub(r'\\n\\n', ' ', ex['text'])
            return ex

        dataset = dataset.map(_map_func)
        dataset = dataset.filter(lambda ex: 0 < len(ex['text'].split()) <= 256)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{text} Is this review positive or negative?", "{answer}"),
            ("{text} What is the sentiment of this review?", "{answer}"),
            ("{text} Was this review given positively or negatively?", "{answer}"),
            ("{text} How would this review be described in terms of sentiment?", "{answer}"),
            ("Is the following review positive or negative? {text}", "{answer}"),
            ("What is the sentiment of the following review? {text}", "{answer}"),
            ("How might one describe the sentiment of this review? {text}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['Negative', 'Positive']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'yelp'
