from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("common_gen")
class Common_gen(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('common_gen', split=split)
        dataset = dataset.map(lambda ex: {'concepts': ", ".join(ex["concepts"])})
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Concepts: {concepts}. Write a sentence that includes all these words.", "{target}"),
            ("Keywords: {concepts}. What is a sentence that includes all these keywords?", "{target}"),
            ("Here are some concepts: {concepts}. What is a sentence about these concepts?", "{target}"),
            ("Produce a sentence which mentions all of these concepts: {concepts}.", "{target}"),
            ("Write a sentence about the following things: {concepts}.", "{target}"),
            ("Generate a sentence that includes all the following words: {concepts}.", "{target}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'rouge'

    @property
    def task_name(self) -> str:
        return 'common_gen'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['target']
