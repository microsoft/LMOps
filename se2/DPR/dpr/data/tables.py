import collections
import csv
import json
import logging
import re
import unicodedata

import jsonlines
import spacy as spacy
from typing import List, Dict


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
console = logging.StreamHandler()
console.setFormatter(log_formatter)

logger.addHandler(console)


class Cell:
    def __init__(self):
        self.value_tokens: List[str] = []
        self.type: str = ""
        self.nested_tables: List[Table] = []

    def __str__(self):
        return " ".join(self.value_tokens)

    def to_dpr_json(self, cell_idx: int):
        r = {"col": cell_idx}
        r["value"] = str(self)
        return r


class Row:
    def __init__(self):
        self.cells: List[Cell] = []

    def __str__(self):
        return "| ".join([str(c) for c in self.cells])

    def visit(self, tokens_function, row_idx: int):
        for i, c in enumerate(self.cells):
            if c.value_tokens:
                tokens_function(c.value_tokens, row_idx, i)

    def to_dpr_json(self, row_idx: int):
        r = {"row": row_idx}
        r["columns"] = [c.to_dpr_json(i) for i, c in enumerate(self.cells)]
        return r


class Table(object):
    def __init__(self, caption=""):
        self.caption = caption
        self.body: List[Row] = []
        self.key = None
        self.gold_match = False

    def __str__(self):
        table_str = "<T>: {}\n".format(self.caption)
        table_str += " rows:\n"
        for i, r in enumerate(self.body):
            table_str += " row #{}: {}\n".format(i, str(r))

        return table_str

    def get_key(self) -> str:
        if not self.key:
            self.key = str(self)
        return self.key

    def visit(self, tokens_function, include_caption: bool = False) -> bool:
        if include_caption:
            tokens_function(self.caption, -1, -1)
        for i, r in enumerate(self.body):
            r.visit(tokens_function, i)

    def to_dpr_json(self):
        r = {
            "caption": self.caption,
            "rows": [r.to_dpr_json(i) for i, r in enumerate(self.body)],
        }
        if self.gold_match:
            r["gold_match"] = 1
        return r


class NQTableParser(object):
    def __init__(self, tokens, is_html_mask, title):
        self.tokens = tokens
        self.is_html_mask = is_html_mask
        self.max_idx = len(self.tokens)
        self.all_tables = []

        self.current_table: Table = None
        self.tables_stack = collections.deque()
        self.title = title

    def parse(self) -> List[Table]:
        self.all_tables = []
        self.tables_stack = collections.deque()

        for i in range(self.max_idx):

            t = self.tokens[i]

            if not self.is_html_mask[i]:
                # cell content
                self._on_content(t)
                continue

            if "<Table" in t:
                self._on_table_start()
            elif t == "</Table>":
                self._on_table_end()
            elif "<Tr" in t:
                self._onRowStart()
            elif t == "</Tr>":
                self._onRowEnd()
            elif "<Td" in t or "<Th" in t:
                self._onCellStart()
            elif t in ["</Td>", "</Th>"]:
                self._on_cell_end()

        return self.all_tables

    def _on_table_start(self):
        caption = self.title
        parent_table = self.current_table
        if parent_table:
            self.tables_stack.append(parent_table)

            caption = parent_table.caption
            if parent_table.body and parent_table.body[-1].cells:
                current_cell = self.current_table.body[-1].cells[-1]
                caption += " | " + " ".join(current_cell.value_tokens)

        t = Table()
        t.caption = caption
        self.current_table = t
        self.all_tables.append(t)

    def _on_table_end(self):
        t = self.current_table
        if t:
            if self.tables_stack:  # t is a nested table
                self.current_table = self.tables_stack.pop()
                if self.current_table.body:
                    current_cell = self.current_table.body[-1].cells[-1]
                    current_cell.nested_tables.append(t)
        else:
            logger.error("table end without table object")

    def _onRowStart(self):
        self.current_table.body.append(Row())

    def _onRowEnd(self):
        pass

    def _onCellStart(self):
        current_row = self.current_table.body[-1]
        current_row.cells.append(Cell())

    def _on_cell_end(self):
        pass

    def _on_content(self, token):
        if self.current_table.body:
            current_row = self.current_table.body[-1]
            current_cell = current_row.cells[-1]
            current_cell.value_tokens.append(token)
        else:  # tokens outside of row/cells. Just append to the table caption.
            self.current_table.caption += " " + token


def read_nq_tables_jsonl(path: str, out_file: str = None) -> Dict[str, Table]:
    tables_with_issues = 0
    single_row_tables = 0
    nested_tables = 0
    regular_tables = 0
    total_tables = 0
    total_rows = 0
    tables_dict = {}

    with jsonlines.open(path, mode="r") as jsonl_reader:
        for jline in jsonl_reader:
            tokens = jline["tokens"]

            if "( hide ) This section has multiple issues" in " ".join(tokens):
                tables_with_issues += 1
                continue
            # if '<Table>' in tokens[1:]:
            #    nested_tables += 1

            mask = jline["html_mask"]
            page_url = jline["doc_url"]
            title = jline["title"]
            # logger.info('Table from page %s', title)
            # logger.info('tokens len %s', len(tokens))
            # logger.info('tokens %s', tokens)
            # logger.info('page_url %s', page_url)
            p = NQTableParser(tokens, mask, title)
            tables = p.parse()

            # logger.info('parsed tables %d', len(tables))

            # table = parse_table(tokens, mask)
            nested_tables += len(tables[1:])

            for t in tables:
                # logger.info('Table: %s', t)
                total_tables += 1

                # calc amount of non empty rows
                non_empty_rows = sum(
                    [
                        1
                        for r in t.body
                        if r.cells and any([True for c in r.cells if c.value_tokens])
                    ]
                )

                if non_empty_rows <= 1:
                    single_row_tables += 1
                else:
                    regular_tables += 1
                    total_rows += len(t.body)

                    if t.get_key() not in tables_dict:
                        tables_dict[t.get_key()] = t

            if len(tables_dict) % 1000 == 0:
                logger.info("tables_dict %d", len(tables_dict))

    print("regular tables", regular_tables)
    print("tables_with_issues", tables_with_issues)
    print("single_row_tables", single_row_tables)
    print("nested_tables", nested_tables)
    if out_file:
        convert_to_csv_for_lucene(tables_dict, out_file)
    return tables_dict


def get_table_string_for_answer_check(table: Table):  # this doesn't use caption
    table_text = ""
    for r in table.body:
        table_text += " . ".join([" ".join(c.value_tokens) for c in r.cells])
    table_text += " . "
    return table_text


def convert_to_csv_for_lucene(tables_dict, out_file: str):
    id = 0
    with open(out_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for _, v in tables_dict.items():
            id += 1
            # strip all
            table_text = get_table_string_for_answer_check(v)
            writer.writerow([id, table_text, v.caption])
    logger.info("Saved to %s", out_file)


def convert_jsonl_to_qas_tsv(path, out):
    results = []
    with jsonlines.open(path, mode="r") as jsonl_reader:
        for jline in jsonl_reader:
            q = jline["question"]
            answers = []
            if "short_answers" in jline:
                answers = jline["short_answers"]

            results.append((q, answers))

    with open(out, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        for r in results:
            writer.writerow([r[0], r[1]])

    logger.info("Saved to %s", out)


nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "ner", "entity_ruler"])


def tokenize(text):
    doc = nlp(text)
    return [token.text.lower() for token in doc]


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize("NFD", text)


def prepare_answers(answers) -> List[List[str]]:
    r = []
    for single_answer in answers:
        single_answer = normalize(single_answer)
        single_answer = single_answer.lower().split(" ")  # tokenize(single_answer)
        r.append(single_answer)
    return r


def has_prepared_answer(prep_answers: List[List[str]], text):
    """Check if a document contains an answer string."""
    text = normalize(text)
    # Answer is a list of possible strings
    text = tokenize(text)
    for single_answer in prep_answers:
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i : i + len(single_answer)]:
                return True
    return False


def has_prepared_answer2(prep_answers: List[List[str]], text: List[str]):
    text = [normalize(token).lower() for token in text]

    # text = [item for sublist in text for item in sublist]

    # text = ' '.join(text)
    # text = normalize(text)
    # text = tokenize(text)

    for single_answer in prep_answers:
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i : i + len(single_answer)]:
                return True
    return False


def has_answer(answers, text, regMatxh=False):
    """Check if a document contains an answer string."""

    text = normalize(text)

    if regMatxh:
        single_answer = normalize(answers[0])
        if regex_match(text, single_answer):
            return True
    else:
        # Answer is a list of possible strings
        text = tokenize(text)

        for single_answer in answers:
            single_answer = normalize(single_answer)
            single_answer = tokenize(single_answer)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True
    return False


def convert_search_res_to_dpr_and_eval(
    res_file, all_tables_file_jsonl, nq_table_file, out_file, gold_res_file: str = None
):
    db = {}
    id = 0
    tables_dict = read_nq_tables_jsonl(all_tables_file_jsonl)
    for _, v in tables_dict.items():
        id += 1
        db[id] = v

    logger.info("db size %s", len(db))
    total = 0
    dpr_results = {}
    import torch

    bm25_per_topk_hits = torch.tensor([0] * 100)
    qas = []
    with open(res_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter="\t")
        # file format: id, text
        for row in reader:
            total += 1
            q = row[0]
            answers = eval(row[1])

            prep_answers = prepare_answers(answers)
            qas.append((q, prep_answers))
            # logger.info('question %s', q)

            question_hns = []
            question_positives = []
            answers_table_links = []

            for k, bm25result in enumerate(row[2:]):
                score, id = bm25result.split(",")
                table = db[int(id)]

                answer_locations = []

                def check_answer(tokens, row_idx: int, cell_idx: int):
                    if has_prepared_answer2(prep_answers, tokens):
                        answer_locations.append((row_idx, cell_idx))

                # logger.info('table %s', table)

                # get string representation to find answer
                if (len(question_positives) >= 10 and len(question_hns) >= 10) or (
                    len(question_hns) >= 30
                ):
                    break

                # table_str = get_table_string_for_answer_check(table)
                table.visit(check_answer)
                has_answer = len(answer_locations) > 0

                if has_answer:
                    # has_answer(answers, table.key)
                    # has_answer(answers, get_table_string_for_answer_check(table))
                    # bm25_per_topk_hits[k:] += 1

                    question_positives.append(table)
                    answers_table_links.append(answer_locations)
                    # break
                else:
                    question_hns.append(table)

            dpr_results[q] = (question_positives, question_hns, answers_table_links)
            if len(dpr_results) % 100 == 0:
                logger.info("dpr_results %s", len(dpr_results))

    logger.info("dpr_results size %s", len(dpr_results))
    logger.info("total %s", total)
    logger.info("bm25_per_topk_hits %s", bm25_per_topk_hits)

    if gold_res_file:
        logger.info("Processing gold_res_file")
        with open(gold_res_file) as cFile:
            csvReader = csv.reader(cFile, delimiter=",")
            for row in csvReader:
                q_id = int(row[0])
                qas_tuple = qas[q_id]
                prep_answers = qas_tuple[1]
                question_gold_positive_match = None
                q = qas_tuple[0]

                # logger.info("q=%s q_id=%s", q, q_id)
                answers_links = None
                for field in row[1:]:
                    psg_id = int(field.split()[0])
                    # logger.info("psg_id=%s", psg_id)
                    # if psg_id >= len(db):
                    #    continue
                    table = db[psg_id]
                    answer_locations = []

                    def check_answer(tokens, row_idx: int, cell_idx: int):
                        if has_prepared_answer2(prep_answers, tokens):
                            answer_locations.append((row_idx, cell_idx))

                    table.visit(check_answer)
                    has_answer = len(answer_locations) > 0
                    if has_answer and question_gold_positive_match is None:
                        question_gold_positive_match = table
                        question_gold_positive_match.gold_match = True
                        answers_links = answer_locations

                if question_gold_positive_match is None:
                    logger.info("No gold match for q=%s, q_id=%s", q, q_id)
                else:  # inject into ctx+ at the first position
                    question_positives, hns, ans_links = dpr_results[q]
                    question_positives.insert(0, question_gold_positive_match)
                    ans_links.insert(0, answers_links)

    # return
    out_results = []
    with jsonlines.open(nq_table_file, mode="r") as jsonl_reader:
        for jline in jsonl_reader:
            q = jline["question"]
            gold_positive_table = jline["contexts"][0]
            mask = gold_positive_table["html_mask"]
            # page_url = jline['doc_url']
            title = jline["title"]
            p = NQTableParser(gold_positive_table["tokens"], mask, title)
            tables = p.parse()
            # select the one with the answer(s)
            prep_answers = prepare_answers(jline["short_answers"])

            tables_with_answers = []
            tables_answer_locations = []

            for t in tables:
                answer_locations = []

                def check_answer(tokens, row_idx: int, cell_idx: int):
                    if has_prepared_answer2(prep_answers, tokens):
                        answer_locations.append((row_idx, cell_idx))

                t.visit(check_answer)
                has_answer = len(answer_locations) > 0
                if has_answer:
                    tables_with_answers.append(t)
                    tables_answer_locations.append(answer_locations)

            if not tables_with_answers:
                logger.info("No answer in gold table(s) for q=%s", q)
                # tables_with_answers.append(tables[0])

            positive_ctxs, hard_neg_ctxs, answers_table_links = dpr_results[q]
            positive_ctxs = positive_ctxs + tables_with_answers
            tables_answer_locations = answers_table_links + tables_answer_locations
            assert len(positive_ctxs) == len(tables_answer_locations)
            positive_ctxs = [t.to_dpr_json() for t in positive_ctxs]

            # set has_answer attributes
            for i, ctx_json in enumerate(positive_ctxs):
                answer_links = tables_answer_locations[i]
                ctx_json["answer_pos"] = answer_links
            hard_neg_ctxs = [t.to_dpr_json() for t in hard_neg_ctxs]
            out_results.append(
                {
                    "question": q,
                    "id": jline["example_id"],
                    "answers": jline["short_answers"],
                    "positive_ctxs": positive_ctxs,
                    "hard_negative_ctxs": hard_neg_ctxs,
                }
            )

    logger.info("out_results size %s", len(out_results))

    with jsonlines.open(
        out_file, mode="w"
    ) as writer:  # encoding="utf-8", .encode('utf-8')
        for r in out_results:
            writer.write(r)

    # with open(out_file, "w") as writer:
    #    writer.write(json.dumps(out_results, indent=4) + "\n")  # indent=4

    logger.info("Saved to %s", out_file)


def convert_long_ans_to_dpr(nq_table_file, out_file):
    out_results = []
    with jsonlines.open(nq_table_file, mode="r") as jsonl_reader:
        for jline in jsonl_reader:
            q = jline["question"]

            gold_positive_table = jline["contexts"]

            mask = gold_positive_table["la_ans_tokens_html_mask"]
            # page_url = jline['doc_url']
            title = jline["title"]

            p = NQTableParser(gold_positive_table["la_ans_tokens"], mask, title)
            tables = p.parse()
            # select the one with the answer(s)

            positive_ctxs = [tables[0].to_dpr_json()]

            out_results.append(
                {
                    "question": q,
                    "id": jline["example_id"],
                    "answers": [],
                    "positive_ctxs": positive_ctxs,
                    "hard_negative_ctxs": [],
                }
            )

    logger.info("out_results size %s", len(out_results))

    with jsonlines.open(
        out_file, mode="w"
    ) as writer:  # encoding="utf-8", .encode('utf-8')
        for r in out_results:
            writer.write(r)

    logger.info("Saved to %s", out_file)


def parse_qa_csv_file(location):
    res = []
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            res.append((question, answers))
    return res


def calc_questions_overlap(tables_file, regular_file, dev_file):
    tab_questions = set()

    with jsonlines.open(tables_file, mode="r") as jsonl_reader:
        logger.info("Reading file %s" % tables_file)
        for jline in jsonl_reader:
            q = jline["question"]
            tab_questions.add(q)

    reg_questions = set()

    if regular_file[-4:] == ".csv":
        qas = parse_qa_csv_file(regular_file)
        for qa in qas:
            reg_questions.add(qa[0])
    else:
        with open(regular_file, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % regular_file)
            data = json.load(f)
            for item in data:
                q = item["question"]
                reg_questions.add(q)
    if dev_file:
        if dev_file[-4:] == ".csv":
            qas = parse_qa_csv_file(dev_file)
            for qa in qas:
                reg_questions.add(qa[0])
        else:
            with open(dev_file, "r", encoding="utf-8") as f:
                logger.info("Reading file %s" % dev_file)
                data = json.load(f)
                for item in data:
                    q = item["question"]
                    reg_questions.add(q)

    logger.info("tab_questions %d", len(tab_questions))
    logger.info("reg_questions %d", len(reg_questions))
    logger.info("overlap %d", len(tab_questions.intersection(reg_questions)))


def convert_train_jsonl_to_ctxmatch(path: str, out_file: str):
    def get_table_string_for_ctx_match(table: dict):  # this doesn't use caption
        table_text = table["caption"] + " . "
        for r in table["rows"]:
            table_text += " . ".join([c["value"] for c in r["columns"]])
        table_text += " . "
        return table_text

    results = []
    with jsonlines.open(path, mode="r") as jsonl_reader:
        for jline in jsonl_reader:
            if len(jline["positive_ctxs"]) == 0:
                continue
            ctx_pos = jline["positive_ctxs"][0]
            table_str = get_table_string_for_ctx_match(ctx_pos)
            q = jline["question"]
            results.append((q, table_str))

            if len(results) % 1000 == 0:
                logger.info("results %d", len(results))

    shards_sz = 3000
    shard = 0

    for s in range(0, len(results), shards_sz):
        chunk = results[s : s + shards_sz]
        shard_file = out_file + ".shard_{}".format(shard)
        with jsonlines.open(shard_file, mode="w") as writer:
            logger.info("Saving to %s", shard_file)
            for i, item in enumerate(chunk):
                writer.write({"id": s + i, "question": item[0], "context": item[1]})
        shard += 1


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None
