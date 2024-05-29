#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to download various preprocessed data sources & checkpoints for DPR
"""

import argparse
import gzip
import logging
import os
import pathlib
import wget

from typing import Tuple

logger = logging.getLogger(__name__)

# TODO: move to hydra config group

NQ_LICENSE_FILES = [
    "https://dl.fbaipublicfiles.com/dpr/nq_license/LICENSE",
    "https://dl.fbaipublicfiles.com/dpr/nq_license/README",
]

RESOURCES_MAP = {
    "data.retriever.break-train-q": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-break-train-q.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA train subset with passages pools for the Retriever training",
    },
    "data.retriever.break-train-qd": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-break-train-qd.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA train subset with passages pools for the Retriever training",
    },
    "data.retriever.break-train-d": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-break-train-d.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA train subset with passages pools for the Retriever training",
    },
    "data.wikipedia_split.psgs_w100": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },
    "data.wikipedia_split.entities": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/entities.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },
    "data.wikipedia_split.break_train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/break_train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Entire wikipedia passages set obtain by splitting all pages into 100-word segments (no overlap)",
    },
    "data.retriever.nq-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "NQ dev subset with passages pools for the Retriever train time validation",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.nq-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "NQ train subset with passages pools for the Retriever training",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.nq-adv-hn-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-adv-hn-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "NQ train subset with hard negative passages mined using the baseline DPR NQ encoders & wikipedia index",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.trivia-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.trivia-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-trivia-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA train subset with passages pools for the Retriever training",
    },
    "data.retriever.grailqa-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-grailqa-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA train subset with passages pools for the Retriever training",
    },
    "data.retriever.grailqa-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-grailqa-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "TriviaQA dev subset with passages pools for the Retriever training",
    },
    "data.retriever.squad1-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "SQUAD 1.1 train subset with passages pools for the Retriever training",
    },
    "data.retriever.squad1-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-squad1-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "SQUAD 1.1 dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.webq-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "WebQuestions dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.webq-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-webquestions-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "WebQuestions dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.curatedtrec-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "CuratedTrec dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.curatedtrec-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-curatedtrec-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "CuratedTrec dev subset with passages pools for the Retriever train time validation",
    },
    "data.retriever.qas.nq-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-dev.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "NQ dev subset for Retriever validation and IR results generation",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.qas.nq-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "NQ test subset for Retriever validation and IR results generation",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever.qas.nq-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/nq-train.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "NQ train subset for Retriever validation and IR results generation",
        "license_files": NQ_LICENSE_FILES,
    },
    #
    "data.retriever.qas.trivia-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-dev.qa.csv.gz",
        "original_ext": ".csv",
        "compressed": True,
        "desc": "Trivia dev subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.break-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/break-dev.qa.jsonl.gz",
        "original_ext": ".jsonl",
        "compressed": True,
        "desc": "Trivia dev subset for Retriever validation and IR results generation",
    },

    "data.retriever.qas.break-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/break-train.qa.jsonl.gz",
        "original_ext": ".jsonl",
        "compressed": True,
        "desc": "Trivia dev subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.grailqa-dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/grailqa-dev.qa.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Trivia dev subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.trivia-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-test.qa.csv.gz",
        "original_ext": ".csv",
        "compressed": True,
        "desc": "Trivia test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.trivia-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/trivia-train.qa.csv.gz",
        "original_ext": ".csv",
        "compressed": True,
        "desc": "Trivia train subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.grailqa-train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/grailqa-train.qa.tsv.gz",
        "original_ext": ".tsv",
        "compressed": True,
        "desc": "Trivia train subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.squad1-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/squad1-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "Trivia test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.webq-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/webquestions-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "WebQuestions test subset for Retriever validation and IR results generation",
    },
    "data.retriever.qas.curatedtrec-test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever/curatedtrec-test.qa.csv",
        "original_ext": ".csv",
        "compressed": False,
        "desc": "CuratedTrec test subset for Retriever validation and IR results generation",
    },
    "data.gold_passages_info.nq_train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-train_gold_info.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our train subset) gold positive passages and alternative question tokenization",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.gold_passages_info.nq_dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-dev_gold_info.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our dev subset) gold positive passages and alternative question tokenization",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.gold_passages_info.nq_test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/nq_gold_info/nq-test_gold_info.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Original NQ (our test, original dev subset) gold positive passages and alternative question "
        "tokenization",
        "license_files": NQ_LICENSE_FILES,
    },
    "pretrained.fairseq.roberta-base.dict": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/pretrained/fairseq/roberta/dict.txt",
        "original_ext": ".txt",
        "compressed": False,
        "desc": "Dictionary for pretrained fairseq roberta model",
    },
    "pretrained.fairseq.roberta-base.model": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/pretrained/fairseq/roberta/model.pt",
        "original_ext": ".pt",
        "compressed": False,
        "desc": "Weights for pretrained fairseq roberta base model",
    },
    "pretrained.pytext.bert-base.model": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/pretrained/pytext/bert/bert-base-uncased.pt",
        "original_ext": ".pt",
        "compressed": False,
        "desc": "Weights for pretrained pytext bert base model",
    },
    "data.retriever_results.nq.single.wikipedia_passages": {
        "s3_url": [
            "https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single/nq/wiki_passages_{}".format(
                i
            )
            for i in range(50)
        ],
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Encoded wikipedia files using a biencoder checkpoint("
        "checkpoint.retriever.single.nq.bert-base-encoder) trained on NQ dataset ",
    },   
    "data.retriever_results.nq.single-adv-hn.wikipedia_passages": {
        "s3_url": [
            "https://dl.fbaipublicfiles.com/dpr/data/wiki_encoded/single-adv-hn/nq/wiki_passages_{}".format(
                i
            )
            for i in range(50)
        ],
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Encoded wikipedia files using a biencoder checkpoint("
        "checkpoint.retriever.single-adv-hn.nq.bert-base-encoder) trained on NQ dataset + adversarial hard negatives",
    },
    "data.retriever_results.nq.single.test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-test.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ test dataset for the encoder trained on NQ",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single.dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-dev.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ dev dataset for the encoder trained on NQ",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single.train": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single/nq-train.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ train dataset for the encoder trained on NQ",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.retriever_results.nq.single-adv-hn.test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/retriever_results/single-adv-hn/nq-test.json.gz",
        "original_ext": ".json",
        "compressed": True,
        "desc": "Retrieval results of NQ test dataset for the encoder trained on NQ + adversarial hard negatives",
        "license_files": NQ_LICENSE_FILES,
    },
    "checkpoint.retriever.single.nq.bert-base-encoder": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriever/single/nq/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on NQ data and HF bert-base-uncased model",
    },
    "checkpoint.retriever.multiset.bert-base-encoder": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/multiset/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on multi set data and HF bert-base-uncased model",
    },
    "checkpoint.retriever.single-adv-hn.nq.bert-base-encoder": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/retriver/single-adv-hn/nq/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Biencoder weights trained on the original DPR NQ data combined with adversarial hard negatives (See data.retriever.nq-adv-hn-train resource). "
        "The model is HF bert-base-uncased",
    },
    "data.reader.nq.single.train": {
        "s3_url": [
            "https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/train.{}.pkl".format(
                i
            )
            for i in range(8)
        ],
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ train dataset input data preprocessed from retriever results (also trained on NQ)",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.reader.nq.single.dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/dev.0.pkl",
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ dev dataset input data preprocessed from retriever results (also trained on NQ)",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.reader.nq.single.test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/reader/nq/single/test.0.pkl",
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model NQ test dataset input data preprocessed from retriever results (also trained on NQ)",
        "license_files": NQ_LICENSE_FILES,
    },
    "data.reader.trivia.multi-hybrid.train": {
        "s3_url": [
            "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/train.{}.pkl".format(
                i
            )
            for i in range(8)
        ],
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia train dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "data.reader.trivia.multi-hybrid.dev": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/dev.0.pkl",
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia dev dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "data.reader.trivia.multi-hybrid.test": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/data/reader/trivia/multi-hybrid/test.0.pkl",
        "original_ext": ".pkl",
        "compressed": False,
        "desc": "Reader model Trivia test dataset input data preprocessed from hybrid retriever results "
        "(where dense part is trained on multiset)",
    },
    "checkpoint.reader.nq-single.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-single/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on NQ-single retriever results and HF bert-base-uncased model",
    },
    "checkpoint.reader.nq-trivia-hybrid.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-trivia-hybrid/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on Trivia multi hybrid retriever results and HF bert-base-uncased model",
    },
    # extra checkpoints for EfficientQA competition
    "checkpoint.reader.nq-single-subset.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-single-seen_only/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on NQ-single retriever results and HF bert-base-uncased model, when only Wikipedia pages seen during training are considered",
    },
    "checkpoint.reader.nq-tfidf.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-drqa/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on TFIDF results and HF bert-base-uncased model",
    },
    "checkpoint.reader.nq-tfidf-subset.hf-bert-base": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/reader/nq-drqa-seen_only/hf_bert_base.cp",
        "original_ext": ".cp",
        "compressed": False,
        "desc": "Reader weights trained on TFIDF results and HF bert-base-uncased model, when only Wikipedia pages seen during training are considered",
    },
    # retrieval indexes
    "indexes.single.nq.full.index": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/full.index.dpr",
        "original_ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever",
    },
    "indexes.single.nq.full.index_meta": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/full.index_meta.dpr",
        "original_ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever (metadata)",
    },
    "indexes.single.nq.subset.index": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/seen_only.index.dpr",
        "original_ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever when only Wikipedia pages seen during training are considered",
    },
    "indexes.single.nq.subset.index_meta": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/single/nq/seen_only.index_meta.dpr",
        "original_ext": ".dpr",
        "compressed": False,
        "desc": "DPR index on NQ-single retriever when only Wikipedia pages seen during training are considered (metadata)",
    },
    "indexes.tfidf.nq.full": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/drqa/nq/full-tfidf.npz",
        "original_ext": ".npz",
        "compressed": False,
        "desc": "TFIDF index",
    },
    "indexes.tfidf.nq.subset": {
        "s3_url": "https://dl.fbaipublicfiles.com/dpr/checkpoint/indexes/drqa/nq/seen_only-tfidf.npz",
        "original_ext": ".npz",
        "compressed": False,
        "desc": "TFIDF index when only Wikipedia pages seen during training are considered",
    },
}


def unpack(gzip_file: str, out_file: str):
    logger.info("Uncompressing %s", gzip_file)
    input = gzip.GzipFile(gzip_file, "rb")
    s = input.read()
    input.close()
    output = open(out_file, "wb")
    output.write(s)
    output.close()
    logger.info(" Saved to %s", out_file)


def download_resource(
    s3_url: str, original_ext: str, compressed: bool, resource_key: str, out_dir: str
) -> Tuple[str, str]:
    logger.info("Requested resource from %s", s3_url)
    path_names = resource_key.split(".")

    if out_dir:
        root_dir = out_dir
    else:
        # since hydra overrides the location for the 'current dir' for every run and we don't want to duplicate
        # resources multiple times, remove the current folder's volatile part
        root_dir = os.path.abspath("./")
        if "/outputs/" in root_dir:
            root_dir = root_dir[: root_dir.index("/outputs/")]

    logger.info("Download root_dir %s", root_dir)

    save_root = os.path.join(
        root_dir, "downloads", *path_names[:-1]
    )  # last segment is for file name

    pathlib.Path(save_root).mkdir(parents=True, exist_ok=True)

    local_file_uncompressed = os.path.abspath(
        os.path.join(save_root, path_names[-1] + original_ext)
    )
    logger.info("File to be downloaded as %s", local_file_uncompressed)

    if os.path.exists(local_file_uncompressed):
        logger.info("File already exist %s", local_file_uncompressed)
        return save_root, local_file_uncompressed

    local_file = os.path.abspath(
        os.path.join(
            save_root, path_names[-1] + (".tmp" if compressed else original_ext)
        )
    )

    wget.download(s3_url, out=local_file)

    logger.info("Downloaded to %s", local_file)

    if compressed:
        uncompressed_file = os.path.join(save_root, path_names[-1] + original_ext)
        unpack(local_file, uncompressed_file)
        os.remove(local_file)
        local_file = uncompressed_file
    return save_root, local_file


def download_file(s3_url: str, out_dir: str, file_name: str):
    logger.info("Loading from %s", s3_url)
    local_file = os.path.join(out_dir, file_name)

    if os.path.exists(local_file):
        logger.info("File already exist %s", local_file)
        return

    wget.download(s3_url, out=local_file)
    logger.info("Downloaded to %s", local_file)


def download(resource_key: str, out_dir: str = None):
    if resource_key not in RESOURCES_MAP:
        # match by prefix
        resources = [k for k in RESOURCES_MAP.keys() if k.startswith(resource_key)]
        if resources:
            for key in resources:
                download(key, out_dir)
        else:
            logger.info("no resources found for specified key")
        return []
    download_info = RESOURCES_MAP[resource_key]

    s3_url = download_info["s3_url"]

    save_root_dir = None
    data_files = []
    if isinstance(s3_url, list):
        for i, url in enumerate(s3_url):
            save_root_dir, local_file = download_resource(
                url,
                download_info["original_ext"],
                download_info["compressed"],
                "{}_{}".format(resource_key, i),
                out_dir,
            )
            data_files.append(local_file)
    else:
        save_root_dir, local_file = download_resource(
            s3_url,
            download_info["original_ext"],
            download_info["compressed"],
            resource_key,
            out_dir,
        )
        data_files.append(local_file)

    license_files = download_info.get("license_files", None)
    if license_files:
        download_file(license_files[0], save_root_dir, "LICENSE")
        download_file(license_files[1], save_root_dir, "README")
    return data_files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir",
        default="./",
        type=str,
        help="The output directory to download file",
    )
    parser.add_argument(
        "--resource",
        type=str,
        help="Resource name. See RESOURCES_MAP for all possible values",
    )
    args = parser.parse_args()
    if args.resource:
        download(args.resource, args.output_dir)
    else:
        print("Please specify resource value. Possible options are:")
        for k, v in RESOURCES_MAP.items():
            print("Resource key=%s  :  %s", k, v["desc"])


if __name__ == "__main__":
    main()
