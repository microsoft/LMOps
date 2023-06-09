import collections
import csv
import json
import logging
import pickle
from typing import Dict

import hydra
import jsonlines
import torch
from omegaconf import DictConfig

from dpr.utils.data_utils import read_data_from_json_files
from dpr.data.biencoder_data import (
    BiEncoderPassage,
    normalize_passage,
    normalize_question,
    get_dpr_files,
    read_nq_tables_jsonl,
    split_tables_to_chunks,
    remove_double_space,
)
from dpr.utils.tasks import (
    task_map,
    get_prompt_files,
)

logger = logging.getLogger(__name__)
QASample = collections.namedtuple(
    "QuerySample", ["query", "id", "answers", "meta_data"]
)
TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])


class RetrieverData(torch.utils.data.Dataset):
    def __init__(self, file: str):
        """
        :param file: - real file name or the resource name as they are defined in download_data.py
        """
        self.file = file
        self.data_files = []

    def load_data(self):
        self.data_files = get_dpr_files(self.file)
        assert (
            len(self.data_files) == 1
        ), "RetrieverData source currently works with single files only. Files specified: {}".format(
            self.data_files
        )
        self.file = self.data_files[0]


class QASrc(RetrieverData):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file)
        self.data = None
        self.selector = hydra.utils.instantiate(selector) if selector else None
        self.special_query_token = special_query_token
        self.query_special_suffix = query_special_suffix

    def __getitem__(self, index) -> QASample:
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _process_question(self, question: str):
        # as of now, always normalize query
        question = normalize_question(question)
        if self.query_special_suffix and not question.endswith(
            self.query_special_suffix
        ):
            question += self.query_special_suffix
        return question


class CsvQASrc(QASrc):
    def __init__(
        self,
        file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_col = question_col
        self.answers_col = answers_col
        self.id_col = id_col

    def load_data(self):
        super().load_data()
        data = []
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                question = row[self.question_col]
                answers = eval(row[self.answers_col])
                id = None
                if self.id_col >= 0:
                    id = row[self.id_col]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class JsonlQASrc(QASrc):
    def __init__(
        self,
        file: str,
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.question_attr = question_attr
        self.answers_attr = answers_attr
        self.id_attr = id_attr

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                answers = jline[self.answers_attr] if self.answers_attr in jline else []
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class KiltCsvQASrc(CsvQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_col: int = 0,
        answers_col: int = 1,
        id_col: int = -1,
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            question_col,
            answers_col,
            id_col,
            selector,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file


class KiltJsonlQASrc(JsonlQASrc):
    def __init__(
        self,
        file: str,
        kilt_gold_file: str,
        question_attr: str = "input",
        answers_attr: str = "answer",
        id_attr: str = "id",
        selector: DictConfig = None,
        special_query_token: str = None,
        query_special_suffix: str = None,
    ):
        super().__init__(
            file,
            selector,
            question_attr,
            answers_attr,
            id_attr,
            special_query_token,
            query_special_suffix,
        )
        self.kilt_gold_file = kilt_gold_file

    def load_data(self):
        super().load_data()
        data = []
        with jsonlines.open(self.file, mode="r") as jsonl_reader:
            for jline in jsonl_reader:
                question = jline[self.question_attr]
                out = jline["output"]
                answers = [o["answer"] for o in out if "answer" in o]
                id = None
                if self.id_attr in jline:
                    id = jline[self.id_attr]
                data.append(QASample(self._process_question(question), id, answers))
        self.data = data


class TTS_ASR_QASrc(QASrc):
    def __init__(self, file: str, trans_file: str):
        super().__init__(file)
        self.trans_file = trans_file

    def load_data(self):
        super().load_data()
        orig_data_dict = {}
        with open(self.file, "r") as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            id = 0
            for row in reader:
                question = row[0]
                answers = eval(row[1])
                orig_data_dict[id] = (question, answers)
                id += 1
        data = []
        with open(self.trans_file, "r") as tfile:
            reader = csv.reader(tfile, delimiter="\t")
            for r in reader:
                row_str = r[0]
                idx = row_str.index("(None-")
                q_id = int(row_str[idx + len("(None-") : -1])
                orig_data = orig_data_dict[q_id]
                answers = orig_data[1]
                q = row_str[:idx].strip().lower()
                data.append(QASample(q, idx, answers))
        self.data = data


class CsvCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        with open(self.file) as ifile:
            reader = csv.reader(ifile, delimiter="\t")
            for row in reader:
                if row[self.id_col] == "id":
                    continue
                if self.id_prefix:
                    sample_id = self.id_prefix + str(row[self.id_col])
                else:
                    sample_id = row[self.id_col]
                passage = row[self.text_col]
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row[self.title_col])


class UpriseQASrc(QASrc):
    def __init__(
        self,
        task_name,
        file="",
        selector: DictConfig = None,
        question_attr: str = "question",
        answers_attr: str = "answers",
        id_attr: str = "id",
        special_query_token: str = None,
        query_special_suffix: str = None,
        cache_dir: str = None,
        task_setup_type: str = "q",
    ):
        super().__init__(file, selector, special_query_token, query_special_suffix)
        self.task = task_map.cls_dic[task_name]()
        logger.info("loading task evaluation split...")
        # load evaluation split defined in task.py
        self.data = self.task.get_dataset( 
            split=None, cache_dir=cache_dir
        )
        self.get_question = self.task.get_question
        assert (
            task_setup_type == "q"
        ), "when testing, the setup should only be q, no answer can be included"

    def load_data(self):
        data = []
        for id, jline in enumerate(self.data):
            jline["id"] = id
            question = self.get_question(jline)
            question = remove_double_space(question)
            answers = ["None"]
            data.append(QASample(question, id, answers, jline))
        self.data = data


def reformat(text):
    return " ".join([f"{i+1}#) {x.strip()}" for i, x in enumerate(text.split(";"))])


class UpriseCtxSrc(RetrieverData):
    def __init__(
        self,
        file="",
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        prompt_pool_path: str = None,
        prompt_setup_type=None,
        train_clusters: str = None,
    ):
        super().__init__(file)
        self.file = file
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.prompt_setup_type = prompt_setup_type
        if train_clusters != None:
            prompt_pool_path = get_prompt_files(prompt_pool_path, train_clusters)
        logger.info("prompt files: %s", prompt_pool_path)
        self.prompt_pool = read_data_from_json_files(prompt_pool_path)
        logger.info("prompt passages num : %d", len(self.prompt_pool))

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        for sample_id, entry in enumerate(self.prompt_pool):
            task = task_map.cls_dic[entry["task_name"]]()
            
            if self.prompt_setup_type == "q":
                passage = task.get_question(entry)
            elif self.prompt_setup_type == "a":
                passage = task.get_answer(entry)
            elif self.prompt_setup_type == "qa":
                passage = (
                    task.get_question(entry)
                    + task.get_answer(entry)
                )
            passage = remove_double_space(passage)
            ctxs[sample_id] = BiEncoderPassage(
                passage, "", entry
            ) # pass the entry as metadata

class JsonCtxSrc(RetrieverData):
    def __init__(
        self,
        file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(file)
        self.text_col = text_col
        self.title_col = title_col
        self.id_col = id_col
        self.id_prefix = id_prefix
        self.normalize = normalize

    def load_data_to(self, ctxs: Dict[object, BiEncoderPassage]):
        super().load_data()
        with open(self.file) as ifile:
            reader = json.load(ifile)
            for row in reader:
                sample_id = row["id"]
                passage = row["text"]
                if self.normalize:
                    passage = normalize_passage(passage)
                ctxs[sample_id] = BiEncoderPassage(passage, row["title"])


class KiltCsvCtxSrc(CsvCtxSrc):
    def __init__(
        self,
        file: str,
        mapping_file: str,
        id_col: int = 0,
        text_col: int = 1,
        title_col: int = 2,
        id_prefix: str = None,
        normalize: bool = False,
    ):
        super().__init__(
            file, id_col, text_col, title_col, id_prefix, normalize=normalize
        )
        self.mapping_file = mapping_file

    def convert_to_kilt(self, kilt_gold_file, dpr_output, kilt_out_file):
        logger.info("Converting to KILT format file: %s", dpr_output)

        with open(dpr_output, "rt") as fin:
            dpr_output = json.load(fin)

        with jsonlines.open(kilt_gold_file, "r") as reader:
            kilt_gold_file = list(reader)
        assert len(kilt_gold_file) == len(dpr_output)
        map_path = self.mapping_file
        with open(map_path, "rb") as fin:
            mapping = pickle.load(fin)

        with jsonlines.open(kilt_out_file, mode="w") as writer:
            for dpr_entry, kilt_gold_entry in zip(dpr_output, kilt_gold_file):
                assert dpr_entry["question"] == kilt_gold_entry["input"]
                provenance = []
                for ctx in dpr_entry["ctxs"]:
                    wikipedia_id, end_paragraph_id = mapping[int(ctx["id"])]
                    provenance.append(
                        {
                            "wikipedia_id": wikipedia_id,
                            "end_paragraph_id": end_paragraph_id,
                        }
                    )
                kilt_entry = {
                    "id": kilt_gold_entry["id"],
                    "input": dpr_entry["question"],
                    "output": [{"provenance": provenance}],
                }
                writer.write(kilt_entry)

        logger.info("Saved KILT formatted results to: %s", kilt_out_file)


class JsonlTablesCtxSrc(object):
    def __init__(
        self,
        file: str,
        tables_chunk_sz: int = 100,
        split_type: str = "type1",
        id_prefix: str = None,
    ):
        self.tables_chunk_sz = tables_chunk_sz
        self.split_type = split_type
        self.file = file
        self.id_prefix = id_prefix

    def load_data_to(self, ctxs: Dict):
        docs = {}
        logger.info("Parsing Tables data from: %s", self.file)
        tables_dict = read_nq_tables_jsonl(self.file)
        table_chunks = split_tables_to_chunks(
            tables_dict, self.tables_chunk_sz, split_type=self.split_type
        )
        for chunk in table_chunks:
            sample_id = self.id_prefix + str(chunk[0])
            docs[sample_id] = TableChunk(chunk[1], chunk[2], chunk[3])
        logger.info("Loaded %d tables chunks", len(docs))
        ctxs.update(docs)
