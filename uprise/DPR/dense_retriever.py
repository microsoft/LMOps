#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import glob
import json
import logging
import pickle
import time
from typing import List, Tuple, Dict, Iterator
import os
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import RepTokenSelector
from dpr.data.qa_validation import calculate_matches, calculate_chunked_matches
from dpr.data.retriever_data import KiltCsvCtxSrc, TableChunk
from dpr.indexer.faiss_indexers import (
    DenseIndexer,
)
from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder, _select_span_with_token
from dpr.options import setup_logger, setup_cfg_gpu, set_cfg_params_from_state
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)

logger = logging.getLogger()
setup_logger(logger)


def generate_question_vectors(
    question_encoder: torch.nn.Module,
    tensorizer: Tensorizer,
    questions: List[str],
    bsz: int,
    query_token: str = None,
    selector: RepTokenSelector = None,
) -> T:
    n = len(questions)
    query_vectors = []

    with torch.no_grad():
        for j, batch_start in enumerate(range(0, n, bsz)):
            batch_questions = questions[batch_start : batch_start + bsz]

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    batch_token_tensors = [
                        _select_span_with_token(q, tensorizer, token_str=query_token)
                        for q in batch_questions
                    ]
                else:
                    batch_token_tensors = [
                        tensorizer.text_to_tensor(" ".join([query_token, q]))
                        for q in batch_questions
                    ]
            else:
                batch_token_tensors = [
                    tensorizer.text_to_tensor(q) for q in batch_questions
                ]

            q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
            q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
            q_attn_mask = tensorizer.get_attn_mask(q_ids_batch)

            if selector:
                rep_positions = selector.get_positions(q_ids_batch, tensorizer)

                _, out, _ = BiEncoder.get_representation(
                    question_encoder,
                    q_ids_batch,
                    q_seg_batch,
                    q_attn_mask,
                    representation_token_pos=rep_positions,
                )
            else:
                _, out, _ = question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

            query_vectors.extend(out.cpu().split(1, dim=0))

            if len(query_vectors) % 100 == 0:
                logger.info("Encoded queries %d", len(query_vectors))

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info("Total encoded queries tensor %s", query_tensor.size())
    assert query_tensor.size(0) == len(questions)
    return query_tensor


class DenseRetriever(object):
    def __init__(
        self, question_encoder: nn.Module, batch_size: int, tensorizer: Tensorizer
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.selector = None

    def generate_question_vectors(
        self, questions: List[str], query_token: str = None
    ) -> T:

        bsz = self.batch_size
        self.question_encoder.eval()
        return generate_question_vectors(
            self.question_encoder,
            self.tensorizer,
            questions,
            bsz,
            query_token=query_token,
            selector=self.selector,
        )


class LocalFaissRetriever(DenseRetriever):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: DenseIndexer,
    ):
        super().__init__(question_encoder, batch_size, tensorizer)
        self.index = index

    def index_encoded_data(
        self,
        vector_files: List[str],
        buffer_size: int,
        path_id_prefixes: List = None,
    ):
        """
        Indexes encoded passages takes form a list of files
        :param vector_files: file names to get passages vectors from
        :param buffer_size: size of a buffer (amount of passages) to send for the indexing at once
        :return:
        """
        buffer = []
        for i, item in enumerate(
            iterate_encoded_files(vector_files, path_id_prefixes=path_id_prefixes)
        ):
            buffer.append(item)
            if 0 < buffer_size == len(buffer):
                self.index.index_data(buffer)
                buffer = []
        self.index.index_data(buffer)
        logger.info("Data indexing completed.")

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        self.index = None
        return results


def validate(
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    return match_stats.questions_doc_hits


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    meta_data,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    # assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        #q_answers = answers[i] 
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(hits)
        meta=meta_data[i]

        merged_data.append(
            {
                "instruction": q,
                #"answers": q_answers,
                "meta_data": meta,
                "ctxs": [
                    {
                        "prompt_pool_id": results_and_scores[0][c], # id in the prompt pool
                        "passage": docs[c][0],
                        "score": scores[c],
                        "meta_data":docs[c][2] 
                    }
                    for c in range(ctxs_num)
                ],
            }
        )

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def iterate_encoded_files(
    vector_files: list, path_id_prefixes: List = None
) -> Iterator[Tuple]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        id_prefix = None
        if path_id_prefixes:
            id_prefix = path_id_prefixes[i]
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                doc = list(doc)
                if id_prefix and not str(doc[0]).startswith(id_prefix):
                    doc[0] = id_prefix + str(doc[0])
                yield doc


def validate_tables(
    passages: Dict[object, TableChunk],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
) -> List[List[bool]]:
    match_stats = calculate_chunked_matches(
        passages, answers, result_ctx_ids, workers_num, match_type
    )
    top_k_chunk_hits = match_stats.top_k_chunk_hits
    top_k_table_hits = match_stats.top_k_table_hits

    logger.info("Validation results: top k documents hits %s", top_k_chunk_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_chunk_hits]
    logger.info("Validation results: top k table chunk hits accuracy %s", top_k_hits)

    logger.info("Validation results: top k tables hits %s", top_k_table_hits)
    top_k_table_hits = [v / len(result_ctx_ids) for v in top_k_table_hits]
    logger.info("Validation results: top k tables accuracy %s", top_k_table_hits)

    return match_stats.top_k_chunk_hits


@hydra.main(config_path="conf", config_name="dense_retriever")
def main(cfg: DictConfig):
    cfg = setup_cfg_gpu(cfg)
    logger.info("CFG (after gpu  configuration):")
    logger.info("%s", OmegaConf.to_yaml(cfg))

    saved_state = load_states_from_checkpoint(cfg.model_file)
    set_cfg_params_from_state(saved_state.encoder_params, cfg)

    tensorizer, encoder, _ = init_biencoder_components(
        cfg.encoder.encoder_model_type, cfg, inference_only=True
    )

    encoder_path = cfg.encoder_path
    if encoder_path:
        logger.info("Selecting encoder: %s", encoder_path)
        encoder = getattr(encoder, encoder_path)
    else:
        logger.info("Selecting standard question encoder")
        encoder = encoder.question_model

    encoder, _ = setup_for_distributed_mode(
        encoder, None, cfg.device, cfg.n_gpu, cfg.local_rank, cfg.fp16
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    logger.info("Loading saved model state ...")

    encoder_prefix = (encoder_path if encoder_path else "question_model") + "."
    prefix_len = len(encoder_prefix)

    logger.info("Encoder state prefix %s", encoder_prefix)
    question_encoder_state = {
        key[prefix_len:]: value
        for (key, value) in saved_state.model_dict.items()
        if key.startswith(encoder_prefix)
    }
    # TODO: long term HF state compatibility fix
    model_to_load.load_state_dict(question_encoder_state, strict=False)
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    questions = []
    question_answers = []
    meta_data=[]
    if not cfg.qa_dataset:
        logger.warning("Please specify qa_dataset to use")
        return

    ds_key = cfg.qa_dataset
    logger.info("qa_dataset: %s", ds_key)

    qa_src = hydra.utils.instantiate(cfg.datasets[ds_key])
    qa_src.load_data()
    for ds_item in qa_src.data:
        question, answers,meta = ds_item.query, ds_item.answers,ds_item.meta_data
        questions.append(question)
        question_answers.append(answers)
        meta_data.append(meta)

    index = hydra.utils.instantiate(cfg.indexers[cfg.indexer])
    logger.info("Index class %s ", type(index))
    index_buffer_sz = index.buffer_size
    index.init_index(vector_size)
    retriever = LocalFaissRetriever(encoder, cfg.batch_size, tensorizer, index)

    logger.info("Using special token %s", qa_src.special_query_token)
    questions_tensor = retriever.generate_question_vectors(
        questions, query_token=qa_src.special_query_token
    )

    if qa_src.selector:
        logger.info("Using custom representation token selector")
        retriever.selector = qa_src.selector

    id_prefixes = []
    ctx_sources = []
    for ctx_src in cfg.ctx_datatsets:
        ctx_src = hydra.utils.instantiate(cfg.ctx_sources[ctx_src])
        id_prefixes.append(ctx_src.id_prefix)
        ctx_sources.append(ctx_src)

    logger.info("id_prefixes per dataset: %s", id_prefixes)

    # index all passages
    ctx_files_patterns = cfg.encoded_ctx_files
    index_path = cfg.index_path

    logger.info("ctx_files_patterns: %s", ctx_files_patterns)
    if ctx_files_patterns:
        assert len(ctx_files_patterns) == len(
            id_prefixes
        ), "ctx len={} pref leb={}".format(len(ctx_files_patterns), len(id_prefixes))
    else:
        assert (
            index_path
        ), "Either encoded_ctx_files or index_path parameter should be set."

    input_paths = []
    path_id_prefixes = []
    for i, pattern in enumerate(ctx_files_patterns):
        pattern_files = glob.glob(pattern)
        pattern_id_prefix = id_prefixes[i]
        input_paths.extend(pattern_files)
        path_id_prefixes.extend([pattern_id_prefix] * len(pattern_files))

    logger.info("Embeddings files id prefixes: %s", path_id_prefixes)

    if index_path and index.index_exists(index_path):
        logger.info("Index path: %s", index_path)
        retriever.index.deserialize(index_path)
    else:
        logger.info("Reading all passages data from files: %s", input_paths)
        retriever.index_encoded_data(
            input_paths, index_buffer_sz, path_id_prefixes=path_id_prefixes
        )
        if index_path:
            retriever.index.serialize(index_path)

    # get top k results
    top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), cfg.n_docs)

    # we no longer need the index
    retriever = None

    all_passages = {}
    for ctx_src in ctx_sources:
        ctx_src.load_data_to(all_passages)

    if len(all_passages) == 0:
        raise RuntimeError(
            "No passages data found. Please specify ctx_file param properly."
        )

    if cfg.validate_as_tables:
        questions_doc_hits = validate_tables(
            all_passages,
            question_answers,
            top_ids_and_scores,
            cfg.validation_workers,
            cfg.match,
        )
    else:
        questions_doc_hits = validate(
            all_passages,
            question_answers,
            top_ids_and_scores,
            cfg.validation_workers,
            cfg.match,
        )

    if cfg.out_file:
        save_results(
            all_passages,
            questions,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            cfg.out_file,
            meta_data,
        )

    if cfg.kilt_out_file:
        kilt_ctx = next(
            iter([ctx for ctx in ctx_sources if isinstance(ctx, KiltCsvCtxSrc)]), None
        )
        if not kilt_ctx:
            raise RuntimeError("No Kilt compatible context file provided")
        assert hasattr(cfg, "kilt_out_file")
        kilt_ctx.convert_to_kilt(qa_src.kilt_gold_file, cfg.out_file, cfg.kilt_out_file)


if __name__ == "__main__":
    main()
