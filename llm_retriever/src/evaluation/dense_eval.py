from typing import List, Dict, Tuple
from datasets import Dataset

from evaluation.base_eval import BaseEval
from config import Arguments
from logger_config import logger


class DenseEval(BaseEval):

    def __init__(self, args: Arguments, corpus: Dataset, **kwargs):
        super().__init__(args, corpus, **kwargs)

        input_prefix = 'query: ' if args.add_qd_prompt else ''
        # TODO: Hack
        is_e5_model = any(e5_name in args.model_name_or_path for e5_name in ['intfloat/e5', 'intfloat/multilingual-e5'])
        if is_e5_model and not input_prefix:
            logger.warning('E5 models need input prefix, set input_prefix = "query: "')
            input_prefix = 'query: '

        from models import SimpleEncoder, SimpleRetriever
        encoder: SimpleEncoder = SimpleEncoder(
            model_name_or_path=args.model_name_or_path,
            l2_normalize=args.l2_normalize,
            prompt=input_prefix,
        )
        cache_dir = '{}/embeddings/'.format(args.output_dir)

        self.retriever: SimpleRetriever = SimpleRetriever(
            encoder=encoder,
            corpus=corpus,
            cache_dir=cache_dir,
        )

    def get_topk_score_doc_ids(self, queries: List[str], k: int, task_names: List[str]) -> List[List[Tuple[float, str]]]:
        assert len(queries) == len(task_names)

        query_idx_to_topk: Dict[int, List[Tuple]] = self.retriever.search_topk(queries=queries, top_k=k)
        for idx in range(len(queries)):
            q_task_name = task_names[idx]
            for j, (score, doc_id) in enumerate(query_idx_to_topk[idx]):
                if str(doc_id) not in self.task_name_to_doc_ids[q_task_name]:
                    query_idx_to_topk[idx][j] = (score - 100., doc_id)
            query_idx_to_topk[idx] = sorted(query_idx_to_topk[idx], key=lambda x: x[0], reverse=True)

        topk_score_doc_ids: List[List[Tuple[float, str]]] = []
        for idx in range(len(queries)):
            score_doc_ids: List[Tuple[float, str]] = query_idx_to_topk[idx][:k]
            score_doc_ids = [(score, str(doc_id)) for score, doc_id in score_doc_ids]
            topk_score_doc_ids.append(score_doc_ids)

        return topk_score_doc_ids
