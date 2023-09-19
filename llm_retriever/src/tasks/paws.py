from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("paws")
class Paws(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('paws', 'labeled_final', split=split)
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{sentence1} {sentence2} Do these sentences mean the same thing?", "{answer}"),
            ("{sentence1} {sentence2} Are these two sentences paraphrases of each other?", "{answer}"),
            ("1. {sentence1} 2. {sentence2} Are these two sentences paraphrases of each other?", "{answer}"),
            ("(1) {sentence1} (2) {sentence2} Do these two sentences mean the same thing?", "{answer}"),
            ("Sentence 1: {sentence1} Sentence 2: {sentence2} Do these two sentences convey the same information?",
             "{answer}"),
            ("Do these two sentences from wikipedia have the same meaning? {sentence1} {sentence2}", "{answer}"),
            ("Same meaning? {sentence1} {sentence2}", "{answer}"),
            ("Are these paraphrases? {sentence1} {sentence2}", "{answer}"),
            ("Do these mean the same? {sentence1} {sentence2}", "{answer}"),
            (
                "Please check if these have the same meaning. Answer \"yes\" if they do, otherwise \"no\". {sentence1} {sentence2}",
                "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['No', 'Yes']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'paws'
