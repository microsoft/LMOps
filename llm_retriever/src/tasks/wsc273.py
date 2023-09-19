from typing import Optional, List, Tuple, Dict, Union
from datasets import load_dataset, Dataset

from tasks import task_map, to_letter
from tasks.base_task import BaseTask


@task_map.add("wsc273")
class Wsc273(BaseTask):
    def _load_raw_data(self, split: str) -> Optional[Dataset]:
        if split == 'train':
            return None

        dataset = load_dataset('winograd_wsc', 'wsc273', split='test')

        def _map_func(ex: Dict) -> Dict:
            text_first = ex["text"][:ex["pronoun_loc"]]
            ex['context'] = text_first

            text_second = ex["text"][ex["pronoun_loc"] + len(ex["pronoun"]):]
            ex['options'] = [ex["options"][0] + text_second, ex["options"][1] + text_second]

            return ex

        dataset = dataset.map(_map_func)

        return dataset

    @property
    def templates(self) -> List[Tuple[str, str]]:
        return [
            ("{context}", "{answer}"),
            ("Complete the passage. {context}", "{answer}"),
            ("How does this following sentence end? {context}", "{answer}"),
            ("What is the most logical completion for the following text? {context}", "{answer}"),
            ("How does this text end? {context}", "{answer}"),
            ("What happens next? {context}", "{answer}"),
            ("Complete the following sentence. {context}", "{answer}"),
            ("Fill in the remainder of the sentence. {context}", "{answer}"),
            ("What is the next event? {context}", "{answer}"),
            ("Complete the rest of the sentence. {context}", "{answer}"),
        ]

    @property
    def possible_answers(self) -> Optional[List[str]]:
        return ['A', 'B']

    @property
    def metric_name(self) -> str:
        return 'simple_accuracy'

    @property
    def task_name(self) -> str:
        return 'wsc273'

    def get_answer(self, example: Dict) -> Union[str, List[str]]:
        return to_letter(example['label'])
