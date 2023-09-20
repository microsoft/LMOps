from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("wsc")
class Wsc(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('super_glue', 'wsc', split=split)

        dataset = dataset.rename_column('text', 'context')
        dataset = dataset.rename_column('span1_text', 'text1')
        dataset = dataset.rename_column('span2_text', 'text2')

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{context} Are \"{text1}\" and \"{text2}\" the same entity?", "{answer}"),
            ("{context} Do \"{text1}\" and \"{text2}\" have the same meaning?", "{answer}"),
            ("Given the following context {context} Are \"{text1}\" and \"{text2}\" the same?", "{answer}"),
            ("{context} Do \"{text2}\" and \"{text1}\" mean the same thing?", "{answer}"),
            ("{context} Are \"{text2}\" and \"{text1}\" the same thing in the aforementioned sentence?", "{answer}"),
            ("Context:{context} Is \"{text2}\" the same as \"{text1}\"?", "{answer}"),
            ("Consider this sentence: {context} Are \"{text2}\" and \"{text1}\" the same?", "{answer}"),
            ("Are \"{text1}\" and \"{text2}\" the same in this sentence? {context}", "{answer}"),
            ("Is \"{text1}\" the same as \"{text2}\" in this sentence? {context}", "{answer}"),
            ("Do \"{text1}\" and \"{text2}\" point to the same thing in the following sentence? {context}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['No', 'Yes']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'wsc'
