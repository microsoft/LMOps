from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("mrpc")
class Mrpc(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('glue', 'mrpc', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Here are two sentences: {sentence1} {sentence2} Do they have the same meaning?", "{answer}"),
            (
                "Here are two sentences: {sentence1} {sentence2} Are the two sentences saying the same thing?",
                "{answer}"),
            ("{sentence1} {sentence2} Do the above sentences mean the same thing?", "{answer}"),
            ("{sentence1} {sentence2} Please tell me if the sentences above mean the same.", "{answer}"),
            ("{sentence1} {sentence2} Are these sentences conveying the same meaning?", "{answer}"),
            ("{sentence1} {sentence2} If the first sentence is true, is the second one also true?", "{answer}"),
            ("{sentence1} {sentence2} Are these two sentences paraphrases of each other?", "{answer}"),
            ("Do the following two sentences have the same meaning? {sentence1} {sentence2}", "{answer}"),
            ("Do these two sentences mean the same thing? {sentence1} {sentence2}", "{answer}"),
            ("Do these sentences have the same meaning? {sentence1} {sentence2}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['No', 'Yes']

    @property
    def metric_name(self) -> str:
        return 'acc_and_f1'

    @property
    def task_name(self) -> str:
        return 'mrpc'
