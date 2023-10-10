import re

from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("e2e_nlg")
class E2e_nlg(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('GEM/e2e_nlg', split=split)

        def _map_func(ex: Dict) -> Dict:
            meaning_representation = re.sub(r'\[', ' = ', ex['meaning_representation'])
            meaning_representation = re.sub(r'\]', '', meaning_representation)
            return {
                'meaning_representation': meaning_representation
            }

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Attributes: {meaning_representation}. Produce a detailed sentence about this restaurant.", "{target}"),
            ("Data: {meaning_representation}. Can you generate a sentence about this data?", "{target}"),
            ("Data: {meaning_representation}. What is a sentence that describe this data?", "{target}"),
            (
                "Here are some keywords about a restaurant: {meaning_representation}. Write a sentence that describes the following attributes of a restaurant.",
                "{target}"),
            (
                "Here is some data about a restaurant: {meaning_representation}. Write a sentence that includes the following data about a restaurant.",
                "{target}"),
            ("Sentence: {meaning_representation}. Can you represent the content in this sentence in data form?",
             "{target}"),
            ("Write a sentence about a restaurant with all the following attributes: {meaning_representation}.",
             "{target}"),
            ("Write a sentence that is about a restaurant with all the following properties: {meaning_representation}.",
             "{target}"),
            ("Produce a detailed sentence about a restaurant using the following words: {meaning_representation}.",
             "{target}"),
            ("Generate a descriptive sentence about a restaurant using the following words: {meaning_representation}.",
             "{target}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'rouge'

    @property
    def task_name(self) -> str:
        return 'e2e_nlg'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['target']
