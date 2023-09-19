from typing import Optional, List, Tuple, Dict
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("sentiment140")
class Sentiment140(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('sentiment140', split=split)

        def _map_func(ex: Dict) -> Dict:
            ex['label'] = 0 if int(ex['sentiment']) == 0 else 1
            return ex

        dataset = dataset.filter(lambda ex: int(ex['sentiment']) in [0, 4])
        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{text} What is the sentiment of this tweet?", "{answer}"),
            ("{text} How would the sentiment of this tweet be described?", "{answer}"),
            ("{text} Describe the sentiment embodied by this tweet.", "{answer}"),
            ("Tweet: {text} Predict the sentiment of this tweet.", "{answer}"),
            ("What is the sentiment of the following tweet? Tweet:{text}", "{answer}"),
            ("How would one describe the sentiment of this tweet? {text}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['Negative', 'Positive']

    @property
    def metric_name(self) -> str:
        # Prior work uses two classes
        # (https://www.aclweb.org/anthology/C14-1008.pdf,
        # https://arxiv.org/pdf/1404.2188.pdf)
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'sentiment140'
