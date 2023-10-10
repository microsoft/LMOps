import os
import tqdm
import torch

from typing import List, Dict, Union
from datasets import Dataset
from collections import defaultdict

from models.simple_encoder import SimpleEncoder
from logger_config import logger


def _sharded_search_topk(
        query_embeds: torch.Tensor, top_k: int,
        shard_embed: torch.Tensor, shard_idx: int,
        idx_offset: int) -> Dict[int, List]:
    query_idx_to_topk: Dict[int, List] = defaultdict(list)
    search_batch_size = 256
    query_indices = list(range(query_embeds.shape[0]))

    for start in tqdm.tqdm(range(0, query_embeds.shape[0], search_batch_size),
                           desc="search shard {}".format(shard_idx),
                           mininterval=5):
        batch_query_embed = query_embeds[start:(start + search_batch_size)]
        batch_query_indices = query_indices[start:(start + search_batch_size)]
        batch_score = torch.mm(batch_query_embed, shard_embed.t())
        batch_sorted_score, batch_sorted_indices = torch.topk(batch_score, k=top_k, dim=-1, largest=True)
        for batch_idx, query_idx in enumerate(batch_query_indices):
            cur_scores = batch_sorted_score[batch_idx].cpu().tolist()
            cur_indices = [str(idx + idx_offset) for idx in batch_sorted_indices[batch_idx].cpu().tolist()]
            query_idx_to_topk[query_idx] += list(zip(cur_scores, cur_indices))
            query_idx_to_topk[query_idx] = sorted(query_idx_to_topk[query_idx], key=lambda t: -t[0])[:top_k]

    return query_idx_to_topk


class SimpleRetriever:

    def __init__(self, encoder: SimpleEncoder,
                 corpus: Union[Dataset, List[str]],
                 cache_dir: str = None):
        self.encoder = encoder

        # Encode the "contents" column of the corpus
        if isinstance(corpus, List):
            corpus = Dataset.from_dict({'contents': corpus})
        self.corpus: Dataset = corpus
        logger.info(f"Corpus size: {len(self.corpus)}")

        self.cache_dir = cache_dir or 'tmp-{}/'.format(len(corpus))
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Cache dir: {self.cache_dir}")
        self.encode_shard_size = 2_000_000

    def search_topk(self, queries: List[str], top_k: int = 10) -> Dict[int, List]:
        # encode the corpus or load from cache if it already exists
        self._encode_corpus_if_necessary(shard_size=self.encode_shard_size)

        # encode the queries
        query_embeds = self._encode_queries(queries)
        if torch.cuda.is_available():
            query_embeds = query_embeds.cuda()

        # search the top-k results
        query_idx_to_topk: Dict[int, List] = defaultdict(list)
        num_shards = (len(self.corpus) + self.encode_shard_size - 1) // self.encode_shard_size
        idx_offset = 0
        for shard_idx in range(num_shards):
            out_path: str = self._get_out_path(shard_idx)
            shard_embeds = torch.load(out_path, map_location=lambda storage, loc: storage)
            shard_embeds = shard_embeds.to(query_embeds.device)
            shard_query_idx_to_topk = _sharded_search_topk(
                query_embeds=query_embeds,
                top_k=top_k,
                shard_embed=shard_embeds,
                shard_idx=shard_idx,
                idx_offset=idx_offset
            )
            for query_idx, shard_topk in shard_query_idx_to_topk.items():
                query_idx_to_topk[query_idx] += shard_topk
                query_idx_to_topk[query_idx] = sorted(query_idx_to_topk[query_idx], key=lambda t: -t[0])[:top_k]

            idx_offset += shard_embeds.shape[0]

        return query_idx_to_topk

    def encode_corpus(self):
        self._encode_corpus_if_necessary(shard_size=self.encode_shard_size)
        logger.info('Done encoding corpus')

    def _get_out_path(self, shard_idx: int) -> str:
        return '{}/shard_{}'.format(self.cache_dir, shard_idx)

    def _encode_corpus_if_necessary(self, shard_size: int):
        num_shards = (len(self.corpus) + shard_size - 1) // shard_size
        num_examples = 0
        for shard_idx in range(num_shards):
            out_path: str = self._get_out_path(shard_idx)
            if os.path.exists(out_path):
                logger.info('{} already exists, will skip encoding'.format(out_path))
                num_examples += len(torch.load(out_path, map_location=lambda storage, loc: storage))
                continue
            shard_dataset: Dataset = self.corpus.shard(
                num_shards=num_shards,
                index=shard_idx,
                contiguous=True
            )
            shard_embeds: torch.Tensor = self.encoder.encode(
                sentences=shard_dataset['contents']
            )

            num_examples += shard_embeds.shape[0]
            logger.info('Saving shard {} ({} examples) to {}'.format(shard_idx, len(shard_dataset), out_path))
            torch.save(shard_embeds, out_path)

        assert num_examples == len(self.corpus), \
            f"Number of examples in the corpus ({len(self.corpus)}) " \
            f"does not match the number of examples in the shards ({num_examples})"

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        return self.encoder.encode(
            sentences=queries
        )
