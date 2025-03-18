import glob
import torch

from typing import List, Dict, Tuple
from datasets import Dataset

from search.simple_encoder import SimpleEncoder
from data_utils import load_corpus
from logger_config import logger


def _get_all_shards_path(index_dir: str) -> List[str]:
    path_list = glob.glob('{}/*-shard-*.pt'.format(index_dir))
    assert len(path_list) > 0

    def _parse_shard_idx(p: str) -> int:
        return int(p.split('-shard-')[1].split('.')[0])

    path_list = sorted(path_list, key=lambda path: _parse_shard_idx(path))
    logger.info('Embeddings path list: {}'.format(path_list))
    return path_list


class E5Searcher:

    def __init__(
            self, index_dir: str,
            model_name_or_path: str = 'intfloat/e5-large-v2',
            verbose: bool = False,
    ):
        self.model_name_or_path = model_name_or_path
        self.index_dir = index_dir
        self.verbose = verbose

        n_gpus: int = torch.cuda.device_count()
        self.gpu_ids: List[int] = list(range(n_gpus))

        self.encoder: SimpleEncoder = SimpleEncoder(
            model_name_or_path=self.model_name_or_path,
            max_length=64,
        )
        self.encoder.to(self.gpu_ids[-1])

        shard_paths = _get_all_shards_path(self.index_dir)
        all_embeddings: torch.Tensor = torch.cat(
            [torch.load(p, weights_only=True, map_location=lambda storage, loc: storage) for p in shard_paths], dim=0
        )
        logger.info(f'Load {all_embeddings.shape[0]} embeddings from {self.index_dir}')

        split_embeddings = torch.chunk(all_embeddings, len(self.gpu_ids))
        self.embeddings: List[torch.Tensor] = [
            split_embeddings[i].to(f'cuda:{self.gpu_ids[i]}', dtype=torch.float16) for i in range(len(self.gpu_ids))
        ]

        self.corpus: Dataset = load_corpus()

    @torch.no_grad()
    def batch_search(self, queries: List[str], k: int, **kwargs) -> List[List[Dict]]:
        query_embed: torch.Tensor = self.encoder.encode_queries(queries).to(dtype=self.embeddings[0].dtype)

        batch_sorted_score, batch_sorted_indices = self._compute_topk(query_embed, k=k)

        results_list: List[List[Dict]] = []
        for query_idx in range(len(queries)):
            results: List[Dict] = []
            for score, idx in zip(batch_sorted_score[query_idx], batch_sorted_indices[query_idx]):
                results.append({
                    'doc_id': int(idx.item()),
                    'score': score.item(),
                })

                if self.verbose:
                    results[-1].update(self.corpus[int(idx.item())])
            results_list.append(results)

        return results_list

    def _compute_topk(self, query_embed: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_score_list: List[torch.Tensor] = []
        batch_sorted_indices_list: List[torch.Tensor] = []

        idx_offset = 0
        for i in range(len(self.embeddings)):
            query_embed = query_embed.to(self.embeddings[i].device)
            score = torch.mm(query_embed, self.embeddings[i].t())
            sorted_score, sorted_indices = torch.topk(score, k=k, dim=-1, largest=True)

            sorted_indices += idx_offset
            batch_score_list.append(sorted_score.cpu())
            batch_sorted_indices_list.append(sorted_indices.cpu())
            idx_offset += self.embeddings[i].shape[0]

        batch_score = torch.cat(batch_score_list, dim=1)
        batch_sorted_indices = torch.cat(batch_sorted_indices_list, dim=1)
        # only keep the top k results based on batch_score
        batch_score, top_indices = torch.topk(batch_score, k=k, dim=-1, largest=True)
        batch_sorted_indices = torch.gather(batch_sorted_indices, dim=1, index=top_indices)

        return batch_score, batch_sorted_indices
