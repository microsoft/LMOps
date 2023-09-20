from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("aeslc")
class Aeslc(BaseTask):

    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        # For some reason, huggingface reports "checksum mismatch", so we ignore the checksum for now
        dataset: Dataset = load_dataset('aeslc', split=split, ignore_verifications=True)
        dataset = dataset.rename_column('email_body', 'body')
        dataset = dataset.rename_column('subject_line', 'subject')

        def _remove_newlines(example: Dict) -> Dict:
            example['body'] = ' '.join(example['body'].split())
            example['subject'] = ' '.join(example['subject'].split())
            return example

        dataset = dataset.map(_remove_newlines, desc='remove newlines')

        # filter logic from uprise. For FLAN, it filters out empty examples
        def _filter_func(example: Dict) -> bool:
            return 0 < len(example['body'].split()) <= 256 and 0 < len(example['subject'].split()) <= 256

        dataset = dataset.filter(_filter_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("What is the subject line for this email? {body}", "{subject}"),
            ("Write a subject line for this message: {body}", "{subject}"),
            ("{body} Write a subject line for this email.", "{subject}"),
            ("Here is an email: {body} What is a potential subject line for this email?", "{subject}"),
            ("{body} Propose a subject line for this email?", "{subject}"),
            ("This is the content of an email: {body} What was the subject line for this email?", "{subject}"),
            ("This is an email {body} What is the subject of this email?", "{subject}"),
            ("{body} Generate a subject line for this email.", "{subject}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'rouge'

    @property
    def task_name(self) -> str:
        return 'aeslc'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['subject']

