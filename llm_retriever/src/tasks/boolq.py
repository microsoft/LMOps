from typing import Optional, List, Tuple
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("boolq")
class Boolq(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'validation'
        dataset = load_dataset('super_glue', 'boolq', split=split)
        dataset = dataset.rename_column('passage', 'text')
        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{text} Can we conclude that {question}?", "{answer}"),
            ("{text} Is it true that {question}?", "{answer}"),
            ("{text} {question}?", "{answer}"),
            ("Text: {text} Question: {question}?", "{answer}"),
            ("{text} What's the best answer to this question: {question}?", "{answer}"),
            ("{text} Based on the above text, what's the best answer to this question: {question}?", "{answer}"),
            ("{text} Answer this question, making sure that the answer is supposed by the text: {question}?",
             "{answer}"),
            ("{text} Is the following statement correct based on the text {question}", "{answer}"),
            ("{text} Is this statement correct \"{question}\"?", "{answer}"),
            ("Is it true that {question} based on the following text? {text}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['No', 'Yes']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'boolq'
