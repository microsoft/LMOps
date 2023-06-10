import hydra
import tqdm
import numpy as np
import json
from rank_bm25 import BM25Okapi
import multiprocessing
from DPR.dpr.utils.tasks import task_map, get_prompt_files
from DPR.dpr.utils.data_utils import read_data_from_json_files
import os
import logging

logger = logging.getLogger(__name__)


class BM25Finder:
    def __init__(self, cfg) -> None:
        self.prompt_setup_type = cfg.prompt_setup_type
        assert self.prompt_setup_type in ["q", "qa", "a"]

        # prompt_pool
        if cfg.train_clusters is not None:
            prompt_pool_path = get_prompt_files(cfg.prompt_pool_path, cfg.train_clusters)
        else:
            prompt_pool_path = cfg.prompt_pool_path
        logger.info("prompt files: %s", prompt_pool_path)
        self.prompt_pool = read_data_from_json_files(prompt_pool_path)
        logger.info("prompt passages num : %d", len(self.prompt_pool))

        logger.info("started creating the corpus")
        self.corpus = [self.tokenize_prompt(prompt) for prompt in self.prompt_pool]
        self.bm25 = BM25Okapi(self.corpus)
        logger.info("finished creating the corpus")

    def tokenize_prompt(self, entry):
        task = task_map.cls_dic[entry["task_name"]]()
        if self.prompt_setup_type == "q":
            prompt = task.get_question(entry)
        elif self.prompt_setup_type == "a":
            prompt = task.get_answer(entry)
        elif self.prompt_setup_type == "qa":
            prompt = (
                task.get_question(entry)
                + " "
                + task.get_answer(entry)
            )
        return self.tokenize(prompt)

    def tokenize(self, text):
        return text.strip().split()

    def detokenize(self, tokens):
        return " ".join(tokens)


def search(tokenized_query, idx, n_docs):
    bm25 = bm25_global
    scores = bm25.get_scores(tokenized_query)
    near_ids = list(np.argsort(scores)[::-1][:n_docs])
    sorted_scores = list(np.sort(scores)[::-1][:n_docs])
    return near_ids, sorted_scores, idx


def _search(args):
    tokenized_query, idx, n_docs = args
    return search(tokenized_query, idx, n_docs)


class GlobalState:
    def __init__(self, bm25) -> None:
        self.bm25 = bm25


def find(cfg):
    finder = BM25Finder(cfg)

    def set_global_object(bm25):
        global bm25_global
        bm25_global = bm25

    pool = multiprocessing.Pool(
        processes=None, initializer=set_global_object, initargs=(finder.bm25,)
    )
    task_name = cfg.task_name
    logger.info("search for %s", task_name)
    task = task_map.cls_dic[task_name]()
    # get the evaluation data split
    dataset = task.get_dataset(cache_dir=cfg.cache_dir)
    get_question = task.get_question
    tokenized_queries = []
    for id, entry in enumerate(dataset):
        entry["id"] = id
        question = get_question(entry)
        tokenized_queries.append(finder.tokenize(question))
    cntx_pre = [
        [tokenized_query, idx, cfg.n_docs]
        for idx, tokenized_query in enumerate(tokenized_queries)
    ]
    cntx_post = {}
    with tqdm.tqdm(total=len(cntx_pre)) as pbar:
        for res in pool.imap_unordered(_search, cntx_pre):
            pbar.update()
            ctx_ids, ctx_scores, idx = res
            cntx_post[idx] = (ctx_ids, ctx_scores)
    merged_data = []
    for idx, data in enumerate(dataset):
        ctx_ids, ctx_scores = cntx_post[idx]
        merged_data.append(
            {
                "instruction": finder.detokenize(tokenized_queries[idx]),
                "meta_data": data,
                "ctxs": [
                    {
                        "prompt_pool_id": str(prompt_pool_id),
                        "passage": finder.detokenize(finder.corpus[prompt_pool_id]),
                        "score": str(ctx_scores[i]),
                        "meta_data": finder.prompt_pool[prompt_pool_id],
                    }
                    for i, prompt_pool_id in enumerate(ctx_ids)
                ],
            }
        )
    with open(cfg.out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores to %s", cfg.out_file)


@hydra.main(config_path="configs", config_name="bm25_retriever")
def main(cfg):
    print(cfg)
    os.makedirs(os.path.dirname(cfg.out_file), exist_ok=True)
    find(cfg)


if __name__ == "__main__":
    main()
