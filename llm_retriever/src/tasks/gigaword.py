from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map
from tasks.base_task import BaseTask


@task_map.add("gigaword")
class Gigaword(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        split = split if split == 'train' else 'test'
        dataset = load_dataset('gigaword', split=split)

        def _filter_func(ex: Dict) -> bool:
            text = ''.join([ex['document'], ex['summary']])
            no_unk = 'UNK' not in text
            no_hashtag = '#' not in text
            return no_unk and no_hashtag

        dataset = dataset.filter(_filter_func)
        dataset = dataset.rename_column('document', 'text')

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("Write a short summary for this text: {text}", "{summary}"),
            ("Briefly summarize this sentence: {text}", "{summary}"),
            ("Generate a short summary this sentence: {text}", "{summary}"),
            ("What is a shorter version of this: {text}", "{summary}"),
            ("{text} Write a brief summary in a sentence or less", "{summary}"),
            ("{text} What is a very short summary of the above text?", "{summary}"),
            ("{text} Summarize the aforementioned text in a single phrase.", "{summary}"),
            ("{text} Can you generate a short summary of the above paragraph?", "{summary}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return None

    @property
    def metric_name(self) -> str:
        return 'rouge'

    @property
    def task_name(self) -> str:
        return 'gigaword'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return example['summary']
