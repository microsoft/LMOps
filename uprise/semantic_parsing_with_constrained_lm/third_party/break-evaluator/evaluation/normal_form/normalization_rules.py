from __future__ import annotations
from typing import Callable
from abc import ABC, abstractmethod

import os
import re
import networkx as nx
import spacy
from spacy.tokens.token import Token
import _pickle as pk
import logging

from evaluation.decomposition import Decomposition

_logger = logging.getLogger(__name__)

class ReferenceToken(object):
    def __init__(self, text, pos="NOUN", tag="NNS", i_min=None, i_max=None):
        self.text = self.lemma_ = text
        self.pos_ = pos
        self.tag_ = tag
        self.i_min = i_min
        self.i_max = i_max

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def contains_index(self, i:int) -> bool:
        return (self.i_min is not None) and (self.i_max is not None) and (self.i_min <= i <= self.i_max)

    def get_id(self):
        return self.get_reference_id(self.text)

    @staticmethod
    def is_reference(text):
        return re.match(r"^@@\d+@@$",text)

    @staticmethod
    def get_reference_id(text):
        return int(text.replace("@", ""))

    @staticmethod
    def from_token(text:str, token:Token):
        rf = ReferenceToken.from_span(text, span=[token])
        return rf

    @staticmethod
    def from_span(text: str, span: [Token | ReferenceToken], **kwargs):
        if not span:
            raise ValueError("span must be non empty tokens list")
        i_min = span[0].i_min if isinstance(span[0], ReferenceToken) else span[0].i
        i_max = span[-1].i_max if isinstance(span[-1], ReferenceToken) else span[-1].i
        rf = ReferenceToken(text, i_min=i_min, i_max=i_max, **kwargs)
        return rf
        # todo: preserved dep_, head_?




cache = {}
def load_cache(path: str) -> {}:
    global cache
    if not os.path.exists(path):
        _logger.warning(f"no available cache:{path}")
        return
    with open(path, 'rb') as f:
        cache = pk.load(f)

def save_cache(path: str):
    if os.path.exists(path):
        _logger.warning(f"already exists cache:{path}")
        return
    with open(path, 'wb') as f:
        pk.dump(cache, f)


def prepare_node(parser, node, mask_references=True):
    def cached_parser(text:str):
        if text in cache:
            return cache[text]
        res = parser(text)
        cache[text] = res
        return res

    clean_label = re.sub(r"\s+", " ", node["label"]).strip().lower()
    doc = cached_parser(clean_label)
    ref_ids = {i: t.text for i, t in enumerate(doc) if ReferenceToken.is_reference(t.text)}

    # replace @@#@@ with "objects" for stable parsing
    if mask_references:
        re_doc = cached_parser(" ".join([("objects" if i in ref_ids else doc[i].text) for i in range(len(doc))]))
        if len(re_doc) == len(doc):
            doc = re_doc

    node["doc"] = [(t if i not in ref_ids else ReferenceToken.from_token(ref_ids[i], t)) for i, t in enumerate(doc)]
    node.pop("label")


class DecomposeRule(ABC):
    def __init__(self):
        self.preserved_tokens = []  # self preserved tokens
        self._preserved_tokens = []  # total preserved tokens on decompose

    def __str__(self):
        return type(self).__name__

    def __repr__(self):
        return self.__str__()

    def decompose(self, node_id: int, graph: nx.DiGraph, preserved_tokens: [str] = None) -> [int]:
        self._preserved_tokens = preserved_tokens and [t for t in preserved_tokens if t not in self.preserved_tokens]
        node = graph.nodes[node_id]
        if "label" in node:
            return False, None
        doc = node["doc"]
        try:
            return self._decompose(node_id=node_id, graph=graph, doc=doc)
        except:
            _logger.exception(f"Decomposition Error: {self._get_doc_str(doc=doc)}")
            return False, None

    @abstractmethod
    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        # todo: doc might contain 'ReferenceToken' which is not token
        raise NotImplementedError

    @staticmethod
    def _get_doc_str(doc: [Token or ReferenceToken]):
        return ' '.join([d.text for d in doc])

    @staticmethod
    def _is_reference(token: Token) -> bool:
        return token.text.startswith("@@")

    @staticmethod
    def _get_reference(node_id: int, span: [Token|ReferenceToken]) -> ReferenceToken:
        return ReferenceToken.from_span(f"@@{node_id}@@", span)

    @staticmethod
    def _add_neighbor(node_id, graph: nx.DiGraph, is_node_to_new: bool = True, **neighbor):
        new_node_id = graph.number_of_nodes()+1
        graph.add_node(new_node_id, **neighbor)
        if is_node_to_new:
            graph.add_edge(node_id, new_node_id)
        else:
            # update predecessors
            for p in graph.predecessors(node_id):
                graph.remove_edge(p, node_id)
                graph.add_edge(p, new_node_id)
            graph.add_edge(new_node_id, node_id)
        return new_node_id

    @staticmethod
    def update_sucessors(graph, n_id, doc:[Token]):
        refs = Decomposition._get_references_ids(" ".join([t.text for t in doc]))
        graph.remove_edges_from([(n_id, s_id) for s_id in graph.successors(n_id)])
        graph.add_edges_from([(n_id, r) for r in refs])

    def extract_spans(self, node_id: int, graph: nx.DiGraph, doc, spans: [(int,int)], is_transactional:bool=False):
        """
        Extract spans (inclusive) from doc to a new node, and point it.
        Extraction is not allowed if:
        * the span is already a reference
        * the span is the entire doc
        * the span violates preserved tokens (see decompose())
        :param node_id:
        :param graph:
        :param doc:
        :param spans:
        :param is_transactional:
        :return: a tuple extracted_refs, new_ids
        extracted_refs: a list of tuples: [ref_value] where its None for no-extracted and ReferenceToken for extracted
        new_ids: new nodes ids
        """
        # todo: distinguish between: cannot extract - its a reference, and cannot extract - its invalidate,
        #  and partial extraction

        if not spans:
            return [],[]

        for start, end in spans:
            if start > end:
                raise ValueError(f"invalid span [{start},{end}]")

        extraction_status = []
        for start, end in spans:
            if (start, end) == (0, len(doc)-1) or not self._validate_preserved(doc=doc, span=(start, end)):
                extraction_status.append([False, None])
                if is_transactional:
                    return [None]*len(spans), []
            elif start == end and isinstance(doc[start], ReferenceToken):
                extraction_status.append([False, doc[start]])
            else:
                extraction_status.append([True, None])  # place holder

        new_doc = []
        new_ids = []
        last_i = 0
        for cur_span, res in zip(spans,extraction_status):
            if not res[0]:
                continue
            new_doc.extend(doc[last_i:cur_span[0]])
            # todo: if span contains reference it should
            neighbor_doc = doc[cur_span[0]:cur_span[1] + 1]
            id = self._add_neighbor(node_id=node_id, graph=graph, doc=neighbor_doc,
                                    is_node_to_new=True)
            self.update_sucessors(graph, id, neighbor_doc)
            new_ids.append(id)
            ref = self._get_reference(id, neighbor_doc)
            res[1] = ref
            new_doc.append(ref)
            last_i = cur_span[1] + 1
        new_doc.extend(doc[last_i:])
        self.update_sucessors(graph, node_id, new_doc)
        graph.nodes[node_id]["doc"] = new_doc
        return [ref for _, ref in extraction_status], new_ids

    def _validate_preserved(self, doc: [Token|ReferenceToken], span:(int, int)) -> bool:
        if not self._preserved_tokens:
            return True
        start, end = span
        span_str = self._get_doc_str(doc[start: end+1])
        for preserved in self._preserved_tokens:
            preserved_strip = preserved.strip()
            if span_str == preserved_strip:
                return False
            preserved_tokens = preserved_strip.split(' ')
            preserved_len = len(preserved_tokens)
            prev_offset = 1 if preserved.startswith(' ') and start>0 else 0  # make sure its not: [preserved ...]
            post_offset = 1 if preserved.endswith(' ') and end+1<len(doc) else 0  # make sure its not: [... preserved]

            """
            preser[ved ...]:
            violate starting at [s-(len-1), s-1] (=> ends at [s-(len-1)+(len-1), s-1+(len-1)])
            => not in [s-(len-1), s+(len-1)-1]
            """
            prev = self._get_doc_str(doc[max(0,start-(preserved_len-1)):
                                         min(len(doc), start+(preserved_len-1) + prev_offset)])

            """
            [... preser]ved:
            violate ending at [e+1, e+(len-1)] (=> starts at [e+1-(len-1), e+(len-1)-(len-1)])
            => not in [e+1-(len-1), e+(len-1)]
            """
            post = self._get_doc_str(doc[max(0,end+1-(preserved_len-1)-post_offset):
                                         min(len(doc),end+preserved_len)])

            if (f" {preserved_strip} " in f" {prev} ") or (f" {preserved_strip} " in f" {post} "):
                return False
        return True

    def _get_test_cases__graphic(self) -> [((int, dict), ([(int, dict)], [(int, int)]))]:
        return []

    def _get_test_cases__str(self) -> (str, [str]):
        return []

    def _test(self):
        parser = spacy.load('en_core_web_sm', disable=['ner'])

        def graphic_test_create_graphs(inp, output: [((int, dict),([(int,dict)], [(int,int)]))]):
            node_id, node_att = inp
            in_g = nx.DiGraph()
            in_g.add_node(node_id, **node_att)

            out_nodes, out_edges = output
            out_g = nx.DiGraph()
            out_g.add_nodes_from(out_nodes)
            out_g.add_edges_from(out_edges)
            return in_g, out_g

        def str_test_create_graphs(inp: str, output:[str]):
            in_g = Decomposition([inp]).to_graph()
            out_g = Decomposition(output).to_graph()
            return in_g, out_g

        def run_test(tests, create_graphs):
            for i, test_case in enumerate(tests):
                inp, output = test_case
                in_g, out_g = create_graphs(inp, output)
                for node in in_g.nodes.values():
                    prepare_node(parser=parser, node=node)
                norm_g = in_g.copy()
                self.decompose(1, norm_g)

                def compare_nodes(n1, n2):
                    if "label" not in n2 and "doc" in n2:
                        n2["label"] = " ".join([t.text for t in n2["doc"]])
                    for k in n1:
                        if k not in n2 or n1[k].lower() != n2[k].lower():
                            return False
                    return True

                def print_case():
                    return "\n".join([
                        f"{str(self)}:",
                        f"test: {list(in_g.nodes.values())}",
                        f"expected: {list(out_g.nodes.values())}",
                        f"norm: {list(norm_g.nodes.values())}"
                    ])

                assert nx.algorithms.is_isomorphic(out_g, norm_g, node_match=compare_nodes), print_case()

        str_tests, graphic_tests = self._get_test_cases__str(), self._get_test_cases__graphic()
        run_test(tests=graphic_tests, create_graphs=graphic_test_create_graphs)
        run_test(tests=str_tests, create_graphs=str_test_create_graphs)


class NounsExtractionDecomposeRule(DecomposeRule):
    def __init__(self):
        super().__init__()

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        noun_components = []
        i = 0
        while i < len(doc):
            while i < len(doc) and not doc[i].tag_.startswith("NN"):
                i += 1
            first = i
            while i < len(doc) and doc[i].tag_.startswith("NN"):
                i += 1
            last = i-1
            if first < last or (first == last and not isinstance(doc[first], ReferenceToken)):
                noun_components.append((first, last))

        # todo: break spans on preserved words? eg: object *blocking* (both are NN, but since blocking is preserved
        #  the extraction fails...)
        _, new_ids = self.extract_spans(node_id=node_id, graph=graph, doc=doc, spans=noun_components)
        return len(new_ids)>0, new_ids

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("blue cubes", ["blue @@2@@", "cubes"]),
            ("cubes", ["cubes"]),
            ("@@1@@", ["@@1@@"]),
            ("@@1@@ in texas", ["@@1@@ in @@2@@", "texas"]),
            ("river in @@1@@", ["@@2@@ in @@1@@", "river"]),
            ("@@1@@ on VLDB conference", ["@@1@@ on @@2@@", "VLDB conference"]),
            #("number of papers", ["number of @@2@@", "papers"]),
        ]


class CompoundNounExtractionDecomposeRule(DecomposeRule):
    """ Assumption: Only nouns """
    def __init__(self):
        super().__init__()

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        # todo: fix func returned value: bool, [int]
        # NN -[compund]-> NN -[compound]-> ...
        if len(doc) < 2:
            return False, None

        for token in doc:
            if not token.tag_.startswith("NN"):
                return False, None
        for token in doc[:-1]:
            if isinstance(token, ReferenceToken) or token.dep_ not in ["compound","amod"]:
                return False, None

        cur_node_id = node_id
        cur_doc = doc
        total_added_ids = []
        while len(cur_doc) > 1:
            extracted_refs, new_ids = self.extract_spans(node_id=cur_node_id, graph=graph, doc=cur_doc,
                                                         spans=[(1,len(cur_doc)-1)], is_transactional=True)
            if None in extracted_refs:
                _logger.warning(f"could not decompose: {cur_doc}")
                break

            total_added_ids.extend(new_ids)
            cur_node_id = extracted_refs[0].get_id()
            cur_doc = cur_doc[1:]
        return len(total_added_ids)>0, total_added_ids

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("gold metal ball", ["gold @@2@@", "metal @@3@@", "ball"]),
            ("gold metal @@1@@", ["gold @@2@@", "metal @@1@@"]),
        ]

class RemoveByConditionDecomposeRule(DecomposeRule):
    def __init__(self, condition: Callable[[Token], bool]):
        super().__init__()
        self.condition = condition

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        to_keep = [token for i,token in enumerate(doc) if (isinstance(token, ReferenceToken) or not self.condition(token)
                                                           or not self._validate_preserved(doc=doc, span=(i,i)))]
        if len(to_keep) < len(doc):
            graph.nodes[node_id]["doc"] = to_keep
            return True, []
        return False, None


class RemoveDETDecomposeRule(RemoveByConditionDecomposeRule):
    def __init__(self):
        super().__init__(condition=lambda token: token.pos_ == "DET" and token.head.lemma_ != "be")

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("the population of @@1@@", ["population of @@1@@"]),
            ("cube that is blue", ["cube that is blue"]),
            ("@@2@@ that contain the keyword Relational Database", ["@@2@@ contain keyword Relational Database"]),
            ("is @@1@@ gold or green", ["is @@1@@ gold or green"]),
            ("is there any @@1@@", ["is there @@1@@"]),
            ("Is there any @@5@@", ["Is there @@5@@"]),
        ]


class AdjectiveDecomposeRule(DecomposeRule):
    def __init__(self):
        super().__init__()

    # todo: should run after extract nouns? doesn't catch multi-noun phrase (e.g New York)
    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        extract_spans = []
        for i,token in enumerate(doc):
            if token.pos_ == "ADJ":
                if i < len(doc)-1 and isinstance(doc[i+1], ReferenceToken):# doc[i+1].tag_.startswith("NN"):
                    extract_spans.append((i,i+1))
                elif i > 1 and doc[i-1].lemma_ == "be" and isinstance(doc[i-2], ReferenceToken): # doc[i-2].tag_.startswith("NN"):
                    # NN is ADJ
                    extract_spans.append((i-2,i))
                elif i > 2 and doc[i-1].lemma_ == "be" and doc[i-2].pos_ == "DET" and isinstance(doc[i-3], ReferenceToken): # doc[i-3].tag_.startswith("NN"):
                    # NN that is ADJ
                    extract_spans.append((i-3,i))
        if not extract_spans:
            return False, None
        _, new_ids = self.extract_spans(node_id=node_id, graph=graph, doc=doc, spans=extract_spans)
        return len(new_ids)>0, new_ids

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("@@1@@ behind the blue @@2@@", ["@@1@@ behind the @@3@@","", "blue @@2@@"]),
            ("@@1@@ behind the @@2@@ that is blue", ["@@1@@ behind the @@3@@","", "@@2@@ that is blue"]),
        ]


class AdjectiveLikeNounDecomposeRule(DecomposeRule):
    def __init__(self):
        super().__init__()

    # todo: should run after extract nouns? doesn't catch multi-noun phrase (e.g New York)
    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        extract_spans = []
        for i,token in enumerate(doc):
            if token.tag_.startswith("NN"):
                if i > 1 and doc[i-1].lemma_ == "be" and isinstance(doc[i-2], ReferenceToken): # doc[i-2].tag_.startswith("NN"):
                    # NN is NN
                    extract_spans.append((i-2,i))
                elif i > 2 and doc[i-1].lemma_ == "be" and doc[i-2].pos_ == "DET" and isinstance(doc[i-3], ReferenceToken): # doc[i-3].tag_.startswith("NN"):
                    # NN that is NN
                    extract_spans.append((i-3,i))
        if not extract_spans:
            return False, None
        _, new_ids = self.extract_spans(node_id=node_id, graph=graph, doc=doc, spans=extract_spans)
        return len(new_ids)>0, new_ids

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("@@1@@ behind the @@2@@ that is metal", ["@@1@@ behind the @@3@@","", "@@2@@ that is metal"]),
            ("if any @@1@@ is gold", ["if any @@2@@", "@@1@@ is gold"]),
            ("if any @@1@@ is @@2@@", ["if any @@3@@","", "@@1@@ is @@2@@"]),
        ]


class ADPDecomposeRule(DecomposeRule):
    def __init__(self):
        super().__init__()

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        if len(doc)>=3 and isinstance(doc[0], ReferenceToken) and doc[1].pos_ in ["ADP", "PART"]:
            end = 3
            while end < len(doc) and doc[end].tag_.startswith("NN"):
                end +=1
            _, new_ids = self.extract_spans(node_id=node_id, graph=graph, doc=doc, spans=[(0,end-1)])
            return len(new_ids) > 0, new_ids
        else:
            return False, None

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("@@1@@ from london to paris", ["@@2@@ to paris", "@@1@@ from london"]),
        ]


def run_tests(cls_root=DecomposeRule):
    import inspect
    def all_subclasses(cls):
        return sorted(set(cls.__subclasses__()).union(
            [s for c in cls.__subclasses__() for s in all_subclasses(c)]), key=lambda c: c.__name__)

    rules = [c() for c in all_subclasses(cls_root) if issubclass(c, ABC) and
             all([(p.default != inspect.Parameter.empty or
                   p.kind in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD])
                  for p in list(inspect.signature(c.__init__).parameters.values())[1:]])]

    for r in rules:
        _logger.info(r)
        r._test()
    _logger.info(f"{len(rules)} rules ended")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    run_tests()
