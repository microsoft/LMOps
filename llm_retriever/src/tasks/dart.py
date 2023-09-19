import re

from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("dart")
class Dart(BaseTask):
    '''
    https://huggingface.co/datasets/GEM/dart
    '''

    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('GEM/dart', split=split)

        def _map_func(ex: Dict) -> Dict:
            tripleset = "; ".join([", ".join(triplet) for triplet in ex["tripleset"]])
            # Get rid of some undesirable cells like "[TABLECONTEXT]", "[TITLE]"
            tripleset = re.sub(r'\[(.*?)\]', '', tripleset)
            return {
                'tripleset': tripleset
            }

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Triple: {tripleset} What is a sentence that describes this triple?", "{target}"),
            ("Data: {tripleset} What would a sentence about this data be like?", "{target}"),
            ("Generate an approximately fifteen-word sentence that describes all this data: {tripleset}", "{target}"),
            ("Here is some data: {tripleset}. Write a sentence that describes this data", "{target}"),
            ("This is some data: {tripleset}. Generate a detailed description of this data", "{target}"),
            ("Generate a sentence about this data: {tripleset}", "{target}"),
            ("Write a sentence that about [{tripleset}].", "{target}"),
            ("Produce a long descriptive sentence that uses all these words: {tripleset}", "{target}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'rouge'

    @property
    def task_name(self) -> str:
        return 'dart'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['target']
