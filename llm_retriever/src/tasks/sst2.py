from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("sst2")
class Sst2(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('sst2', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Review: \"{sentence}\" Is this movie review sentence negative or positive?", "{answer}"),
            ("Short movie review: \"{sentence}\" Did the critic thinking positively or negatively of the movie?",
             "{answer}"),
            (
                "Sentence from a movie review: \"{sentence}\" Was the movie seen positively or negatively based on the preceding review?",
                "{answer}"),
            ("\"{sentence}\" How would the sentiment of this sentence be perceived?", "{answer}"),
            ("Is the sentiment of the following sentence positive or negative? \"{sentence}\"", "{answer}"),
            ("What is the sentiment of the following movie review sentence? \"{sentence}\"", "{answer}"),
            ("Would the following phrase be considered positive or negative? \"{sentence}\"", "{answer}"),
            ("Does the following review have a positive or negative opinion of the movie? \"{sentence}\"", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['Negative', 'Positive']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'sst2'
