from abc import abstractmethod
from typing import List, Dict, Set, Tuple
from collections import defaultdict
from datasets import Dataset

from config import Arguments


class BaseEval:

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        self.args: Arguments = args
        # id / contents / task_name
        self.corpus: Dataset = corpus

        self.task_name_to_doc_ids: Dict[str, Set[str]] = defaultdict(set)
        for doc_id, task_name in zip(self.corpus['id'], self.corpus['task_name']):
            self.task_name_to_doc_ids[task_name].add(doc_id)

    @abstractmethod
    def get_topk_score_doc_ids(
            self, queries: List[str], k: int, task_names: List[str]
    ) -> List[List[Tuple[float, str]]]:
        raise NotImplementedError

    def get_doc_ids_by_task_name(self, task_name: str) -> List[str]:
        return list(self.task_name_to_doc_ids[task_name])

    def get_prompt_by_doc_ids(self, doc_ids: List[str]) -> str:
        return '\n\n'.join([self.corpus[int(doc_id)]['contents'] for doc_id in doc_ids])
