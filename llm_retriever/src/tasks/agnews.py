from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("ag_news")
class Ag_news(BaseTask):

    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('ag_news', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("\"{text}\" What is this text about? World, Sports, Business, or Technology?", "{answer}"),
            ("\"{text}\" Which topic is this article about? World, Sports, Business, or Technology?", "{answer}"),
            ("\"{text}\" Which is the best summary of this article? World, Sports, Business, or Technology?",
             "{answer}"),
            ("\"{text}\" What is this text about? World, Sports, Business, or Technology?", "{answer}"),
            (
                "\"{text}\" What best summarizes the content of the above article? World, Sports, Business, or Technology?",
                "{answer}"),
            ("Which is this about? \"{text}\" World, Sports, Business, or Technology?", "{answer}"),
            ("Which is an appropriate title for this article? \"{text}\" World, Sports, Business, or Technology?",
             "{answer}"),
            ("Select the topic that this about: \"{text}\" World, Sports, Business, or Technology?", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['World', 'Sports', 'Business', 'Technology']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'ag_news'
