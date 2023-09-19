import random

from typing import List, Tuple, Dict
from datasets import Dataset

from evaluation.base_eval import BaseEval
from config import Arguments
from logger_config import logger


# This class randomly selects k-shot examples from the training set
class RandomEval(BaseEval):

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        super().__init__(args, corpus, **kwargs)
        self.all_doc_ids: List[str] = self.corpus['id']
        self.cached_task_name_to_doc_ids: Dict[str, List[str]] = {}

    def get_topk_score_doc_ids(self, queries: List[str], k: int, task_names: List[str]) -> List[List[Tuple[float, str]]]:
        assert len(queries) == len(task_names)

        topk_score_doc_ids: List[List[Tuple[float, str]]] = []
        for query, task_name in zip(queries, task_names):
            random_score_doc_ids: List[str] = self._single_get_topk_doc_ids(query, k, task_name)
            topk_score_doc_ids.append([(-1, doc_id) for doc_id in random_score_doc_ids])

        return topk_score_doc_ids

    def _single_get_topk_doc_ids(self, query: str, k: int, task_name: str) -> List[str]:
        if task_name not in self.cached_task_name_to_doc_ids:
            self.cached_task_name_to_doc_ids[task_name] = self.get_doc_ids_by_task_name(task_name)
        doc_ids: List[str] = self.cached_task_name_to_doc_ids[task_name]
        # mnli_m & mnli_mm should retrieve from mnli training set
        if len(doc_ids) == 0 and task_name.startswith('mnli_'):
            if 'mnli' not in self.cached_task_name_to_doc_ids:
                self.cached_task_name_to_doc_ids['mnli'] = self.get_doc_ids_by_task_name('mnli')
            doc_ids = self.cached_task_name_to_doc_ids['mnli']

        if len(doc_ids) == 0:
            logger.warning('Use the whole training set for task: {}'.format(task_name))
            doc_ids = self.all_doc_ids

        if k >= len(doc_ids):
            logger.warning('k ({}) is larger than the number of examples ({})'.format(k, len(doc_ids)))
        k = min(k, len(doc_ids))

        return random.sample(doc_ids, k)
