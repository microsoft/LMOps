from abc import ABC
import logging

import networkx as nx
from spacy.tokens.token import Token

from scripts.qdmr_to_program import QDMROperation
import scripts.qdmr_to_program as qdmr
from evaluation.normal_form.normalization_rules import DecomposeRule, ReferenceToken, run_tests

_logger = logging.getLogger(__name__)


class OperationDecomposeRule(DecomposeRule, ABC):
    def __init__(self, is_extract_params=False):
        super().__init__()
        self.operation:QDMROperation = None
        self.is_extract_params = is_extract_params

    def update_node(self, node_id, graph, params_span:[(int, int)], meta:[str]=[], keep_order: bool=True):
        node = graph.nodes[node_id]

        for start,end in params_span:
            if not self._validate_preserved(doc=node["doc"], span=(start, end)):
                return False, None

        if self.is_extract_params:
            extracted_refs, new_ids = self.extract_spans(node_id=node_id, graph=graph, doc=node["doc"], spans=params_span, is_transactional=True)
            if None in extracted_refs:
                return False, None
            self._update_node(node=node, params=[[x] for x in extracted_refs], meta=meta)
            return True, new_ids or []

        self._update_node(node=node, params=[node["doc"][s:e+1] for (s,e) in params_span], meta=meta, keep_order=keep_order)
        return True, []

    def _update_node(self, node, params:[[Token]], meta:[str]=[], keep_order: bool=True):
        params = [' '.join([t.lemma_ for t in param]) for param in params]
        if not keep_order:
            params = sorted(params)
        node["label"] = f"{self.operation.name}{'['+','.join(meta)+']' if meta else ''}({','.join(params)})"
        node["operation"] = self.operation
        node["meta"] = meta

    @staticmethod
    def _is_contains_index(token: Token or ReferenceToken, index:int):
        if isinstance(token, ReferenceToken):
            return token.contains_index(index)
        return token.i == index


class AggregateDecomposeRule(OperationDecomposeRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.AGGREGATE
        self.metadata_map = {
            'COUNT': ['number of'],
            'MAX': ['biggest', 'higher', 'highest', 'larger', 'largest', 'last', 'longer', 'longest',
                    'max', 'maximum', 'more', 'most'],
            'MIN': ['fewer', 'least', 'less', 'lower', 'lowest', 'min', 'minimum', 'shortest', 'smaller', 'smallest'],
            'SUM': ['sum', 'total'],
            'AVG': ['average', 'avg', 'mean'],
        }
        self._spans_to_metadata = {s:meta for meta,spans in self.metadata_map.items() for s in spans}

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        doc_len = len(doc)

        meta = None
        rest_start_index= None

        if doc_len >= 3 and not isinstance(doc[0], ReferenceToken) and doc[0].dep_ == "ROOT" and doc[1].lemma_ == "of":
            # ROOT[number (of), sum, mean] of <...>  => AG(@@#@@); <...>
            meta = self._spans_to_metadata.get(doc[0].text, None) or self._spans_to_metadata.get(f"{doc[0].text} of", None)
            if meta:
                rest_start_index = 2
        if not meta and doc_len>=2 and (doc[0].text in self._spans_to_metadata) and \
                ((doc[1].tag_.startswith("NN")) or (doc[1].lemma_ == "of" and doc_len>=3 and doc[2].tag_.startswith("NN"))):
            # ADJ[MAX, MIN, total] (of) ROOT = > AG(@@#@@); <...>
            meta = self._spans_to_metadata[doc[0].text]
            rest_start_index = 1 if doc[1].lemma_ != "of" else 2

        if meta and rest_start_index:
            return self.update_node(node_id=node_id, graph=graph, params_span=[(rest_start_index, doc_len-1)], meta=[meta])
        return False, None

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("number of @@1@@", ["AGGREGATE[COUNT](@@1@@)"]),
            ("largest balls", ["AGGREGATE[MAX](ball)"]),
        ]


class FilterAdjectiveDecomposeRule(OperationDecomposeRule):
    """ Assumption: first parameter is ReferenceToken """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.FILTER

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        adj_index, nn_index = -1,-1
        if len(doc) == 2 and doc[0].pos_ == "ADJ" and isinstance(doc[1],ReferenceToken):
            # ADJ NN
            adj_index, nn_index = 0, 1
        elif len(doc) == 3 and isinstance(doc[0], ReferenceToken) and doc[1].lemma_ == "be" and doc[2].pos_ == "ADJ":
            # NN is ADJ
            nn_index, adj_index = 0, 2
        elif len(doc) == 4 and isinstance(doc[0], ReferenceToken) and doc[1].pos_ == "DET" and doc[2].lemma_ == "be" and doc[3].pos_ == "ADJ":
            # NN that be ADJ
            nn_index, adj_index = 0, 3

        if adj_index >= 0 and nn_index >= 0:
            return self.update_node(node_id=node_id, graph=graph, params_span=[(nn_index,nn_index), (adj_index,adj_index)])
        return False, None

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("blue cubes", ["blue cubes"]),
            ("blue @@1@@", ["FILTER(@@1@@,blue)"]),
            ("@@1@@ is blue", ["FILTER(@@1@@,blue)"]),
            ("@@1@@ that are blue", ["FILTER(@@1@@,blue)"]),
        ]


class FilterAdjectiveLikeNounDecomposeRule(OperationDecomposeRule):
    """ Assumption: first parameter is ReferenceToken """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.FILTER

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        adj_index, nn_index = -1,-1
        if len(doc) == 2 and isinstance(doc[0], ReferenceToken) and doc[1].pos_ == "NOUN":
            # NN NN
            nn_index, adj_index = 0, 1
        if len(doc) == 3 and isinstance(doc[0], ReferenceToken) and doc[1].lemma_ == "be" and doc[2].pos_ == "NOUN":
            # NN is NN
            nn_index, adj_index = 0, 2
        elif len(doc) == 4 and isinstance(doc[0], ReferenceToken) and doc[1].pos_ == "DET" and doc[2].lemma_ == "be" and doc[3].pos_ == "NOUN":
            # NN that be NN
            nn_index, adj_index = 0, 3

        if adj_index >= 0 and nn_index >= 0:
            return self.update_node(node_id=node_id, graph=graph, params_span=[(nn_index, nn_index), (adj_index, adj_index)])
        return False, None

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("@@1@@ is metal", ["FILTER(@@1@@,metal)"]),
            ("@@1@@ that are metal", ["FILTER(@@1@@,metal)"]),
            ("@@1@@ matte", ["FILTER(@@1@@,matte)"]),
        ]


class FilterADPDecomposeRule(OperationDecomposeRule):
    """ Assumption: first parameter is ReferenceToken """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.FILTER

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        if len(doc) == 3 and isinstance(doc[0], ReferenceToken) and doc[1].pos_ in ["ADP", "PART"]:
            return self.update_node(node_id=node_id, graph=graph, params_span=[(0, 0), (1, 2)])
        return False, None

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("@@1@@ to london", ["FILTER(@@1@@,to london)"]),
        ]


class FilterCompoundNounDecomposeRule(OperationDecomposeRule):
    """ Assumption: Only nouns """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.FILTER

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        # todo: fix func returned value: bool, [int]
        # NN -[compund]-> NN -(ref)
        if len(doc) != 2:
            return False, None

        if doc[0].tag_.startswith("NN") and isinstance(doc[1], ReferenceToken): # todo: and doc[0].dep_ in ["compound","amod"]:
            return self.update_node(node_id=node_id, graph=graph, params_span=[(1, 1), (0, 0)])
        return False, None

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("gold @@1@@", ["FILTER(@@1@@,gold)"]),
        ]


class FilterConditionDecomposeRule(OperationDecomposeRule):
    """ Assumption: first parameter is ReferenceToken """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.FILTER

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        # @@#@@@(NN) [that] [is] ... VERB ...
        if not doc or not isinstance(doc[0], ReferenceToken):
            return False, None
        verb_index = None
        for i, t in enumerate(doc):
            if t.pos_ == "VERB" and t.lemma_ != "be" and doc[0].contains_index(t.head.i):
                verb_index = i
                break
        if not verb_index:
            return False, None
        end_index = verb_index+1
        while end_index < len(doc) and (isinstance(doc[end_index], ReferenceToken) or doc[verb_index].is_ancestor(doc[end_index])):
            # todo: check if VERB is ancestor of ReferenceToken
            end_index += 1
        end_index -= 1
        if not end_index == len(doc)-1:
            return False, None

        condition_start = 1
        if condition_start + 1 < len(doc) and doc[condition_start].lemma_ == "that" and doc[condition_start + 1].lemma_ == "be":
            condition_start += 2
        elif condition_start < len(doc) and (doc[condition_start].lemma_ in ["that", "be"]):
            condition_start += 1

        return self.update_node(node_id=node_id, graph=graph, params_span=[(0, 0), (condition_start, end_index+1)])

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("@@1@@ that is partially hidden by a ball", ["FILTER(@@1@@,partially hide by a ball)"]),
            ("@@1@@ partially hidden by a ball", ["FILTER(@@1@@,partially hide by a ball)"]),
            ("@@1@@ that contain the keyword Relational Database", ["FILTER(@@1@@,contain the keyword Relational Database)"]),
        ]


class SelectionDecomposeRule(OperationDecomposeRule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.SELECT

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        node = graph.nodes[node_id]
        if len(doc) == 1:  # single noun
            token = doc[0]
            if token.pos_ != "NOUN" or isinstance(token, ReferenceToken):
                return False, None
        else:   # noun phrase
            if not all(t.pos_ == "NOUN" for t in doc) or len([t for t in doc if isinstance(t, ReferenceToken) or t.dep_ != "compound"])>1:
                return False, None

        return self.update_node(node_id=node_id, graph=graph, params_span=[(0, len(doc)-1)])

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("cubes",["SELECT(cube)"]),
            ("University of Michigan", ["University of Michigan"]),
            ("VLDB conference", ["SELECT(vldb conference)"]),
        ]

    def _get_test_cases__graphic(self) -> [((int, dict), [(int, int, dict)])]:
        return [
            ((1, {"label": "cubes"}), ([(1, {"label": "SELECT(cube)"})], [])),
            ((1, {"label": "@@1@@"}), ([(1, {"label": "@@1@@"})], [])),
        ]


class WrapperDecomposeRule(OperationDecomposeRule):
    _preserved_tokens_map = {
        QDMROperation.AGGREGATE:
            [
                'number of ', 'highest', 'largest', 'lowest', 'smallest', 'maximum', 'minimum', 'max', 'min', 'sum',
                'total', 'average ', 'avg ', 'mean ',
                'most', 'longest', 'biggest', 'more', 'last', 'longer', 'higher',
                'larger', 'smallest', 'least', 'shortest', 'less', 'first', 'shorter', 'lower', 'fewer',
                'smaller', 'true ', 'false '
            ],
        QDMROperation.GROUP:
            [
                'for each '
            ],
        QDMROperation.COMPARATIVE:
            [
                'same as ', 'higher than ', 'larger than ', 'smaller than ', 'lower than ', 'more than ', 'less than ',
                'more', 'less', 'at least', 'at most', 'equal', 'contain ', 'include ', 'has ', 'have ', 'end with ',
                'start with ', 'ends with ', 'starts with ', 'begin',
                'higher', 'larger', 'smaller', 'lower', 'not ', 'same as', 'the same',
                'equal to ', 'where '
            ],
        QDMROperation.SUPERLATIVE:
            [
                'highest', 'largest', 'most', 'smallest', 'lowest', 'smallest', 'least', 'longest', 'shortest',
                'biggest'
            ],
        QDMROperation.UNION:
            [
                ' and ', ' or ', ' , '
            ],
        QDMROperation.INTERSECTION:
            [
                'both', ' and ', 'of both ', 'in both ', 'by both '
            ],
        QDMROperation.DISCARD:
            [
                'besides ', 'not in '
            ],
        QDMROperation.SORT:
            [
                'sorted by ', 'order by ', 'ordered by '
            ],
        QDMROperation.BOOLEAN:
            [
                'if ', 'is ', 'are ', ' any'
            ],
        QDMROperation.ARITHMETIC:
            [
                'sum', 'difference', 'multiplication', 'division'
            ],
        QDMROperation.FILTER:
            [
                # positional
                ' left ', ' right ', ' between ', ' behind ', ' in front ', ' infront ', ' touch', ' reflect', ' cover',
                ' obscur', ' blocking', ' blocked', ' hidden', ' obstruct', ' near', ' next', ' next to ', ' close ',
                ' closer ', ' closest ', ' adjacent '
            ] + [
                # filter
                ' that is ', ' that are '
            ]
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = None
        self.preserved_tokens = [i for v in self._preserved_tokens_map.values() for i in v]

    @staticmethod
    def fix_preserved_tokens(preserved_tokens, operation):
        if preserved_tokens is not None:
            not_allowed = [t for k, v in WrapperDecomposeRule._preserved_tokens_map.items() for t in v if k != operation]
            allowed = WrapperDecomposeRule._preserved_tokens_map.get(operation, [])
            preserved_tokens = [x for x in (preserved_tokens + not_allowed) if x not in allowed]
        return preserved_tokens

    class _Dummy(dict):
        def __getitem__(self, key):
            return f"#{key}"

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        try:
            qdmr_step = ' '.join([(f"#{t.get_id()}" if isinstance(t, ReferenceToken) else t.text) for t in doc])
            self.operation = qdmr.step_type(qdmr_step, is_high_level=False)  # todo: deal with high_level
            if self.operation in [QDMROperation.NONE, QDMROperation.SELECT]:
                return False, None

            prev_refs_code = self._Dummy()  # dummy values - keep references the same
            exec_qdmr = qdmr.ExecQDMR(self.operation, qdmr_step, prev_refs_code)

            if not exec_qdmr.arguments:
                return False, None

            # align args to doc
            arguments_spans = []
            meta = []
            # todo: # => @@#@@
            for arg in exec_qdmr.arguments:
                arg = qdmr.qdmr_to_prediction(arg)
                arg_tokens = arg.split(' ')
                start_index = 0
                span = None
                while start_index + len(arg_tokens) <= len(doc):
                    if all([at == dt.text for at,dt in zip(arg_tokens, doc[start_index:start_index+len(arg_tokens)])]):
                        span = (start_index, start_index+len(arg_tokens)-1)
                        break
                    start_index += 1
                if span:
                    arguments_spans.append(span)
                else:
                    meta.append(arg)

            if self.operation == QDMROperation.AGGREGATE:
                assert len(meta) == 1 and len(arguments_spans) == 1, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.BOOLEAN:
                assert len(meta) == 1 and len(arguments_spans) >= 1, f"unexpected args parse {len(meta), len(arguments_spans)}"
                # todo: elaborate
            elif self.operation == QDMROperation.COMPARISON:
                assert len(meta) == 1 and len(arguments_spans) >= 2, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.COMPARATIVE:
                assert len(meta) == 1 and len (arguments_spans) == 3, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.DISCARD:
                assert len(meta) == 0 and len(arguments_spans) == 2, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.FILTER:
                assert len(meta) <= 1 and len(arguments_spans) == 2, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.GROUP:
                assert len(meta) == 1 and len(arguments_spans) == 1, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.INTERSECTION:
                assert len(meta) == 0 and len(arguments_spans) >= 3, f"unexpected args parse {len(meta), len(arguments_spans)}"
            # elif self.operation == QDMROperation.PROJECT:
            #     # project is buggy right now.
            #     assert len(meta) == 0 and len(arguments_spans) == 2, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.SORT:
                assert len(meta) == 0 and len(arguments_spans) == 2, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.SUPERLATIVE:
                assert len(meta) == 1 and len(arguments_spans) == 2, f"unexpected args parse {len(meta), len(arguments_spans)}"
            elif self.operation == QDMROperation.UNION:
                assert len(meta) == 0 and len(arguments_spans) >= 2, f"unexpected args parse {len(meta), len(arguments_spans)}"
            else:
                return False, None

            # fix preserved
            self._preserved_tokens = self.fix_preserved_tokens(preserved_tokens=self._preserved_tokens,
                                                               operation=self.operation)

            return self.update_node(node_id=node_id, graph=graph, params_span=arguments_spans, meta=meta)
        except Exception as ex:
            _logger.debug(self._get_doc_str(doc=doc), exc_info=True)
            return False, None

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("the number of @@1@@", ["AGGREGATE[COUNT](@@1@@)"]),
            ("if @@1@@ be the same as @@2@@", ["BOOLEAN[=](@@1@@,@@2@@)"]),
            ("which is more @@1@@ or @@2@@", ["COMPARISON[MAX](@@1@@,@@2@@)"]),
            ("@@2@@ besides @@1@@", ["DISCARD(@@2@@,@@1@@)"]),
            ("@@1@@ that is partially hidden by @@2@@", ["FILTER[POS_COVERS](@@1@@,@@2@@)"]),
            ("@@1@@ that is left of @@2@@", ["FILTER[POS_LEFT_OF](@@1@@,@@2@@)"]),
            # ("colors of @@1@@", ["PROJECT(color,@@1@@)"]),
            ("both @@1@@ and @@2@@", ["UNION(@@1@@,@@2@@)"]),
        ]


class WrapperFixesAggregateDecomposeRule(OperationDecomposeRule):
    """ Assume to be run before wrapper """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.AGGREGATE
        self.preserved_tokens = WrapperDecomposeRule._preserved_tokens_map.get(self.operation, [])

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        if not doc:
            return False, None

        arguments_spans = []
        meta = []
        if len(doc)>=4 and isinstance(doc[0], ReferenceToken) and doc[1].lemma_ == "that" and doc[2].lemma_ == "be"  \
                and " ".join(x.text for x in doc[3:]) in WrapperDecomposeRule._preserved_tokens_map.get(QDMROperation.AGGREGATE, []):
            arguments_spans = [(0, 0)]
        elif len(doc)>=2 and isinstance(doc[-1], ReferenceToken) and " ".join(x.text for x in doc[:-1]) in WrapperDecomposeRule._preserved_tokens_map.get(QDMROperation.AGGREGATE, []):
            arguments_spans = [(len(doc)-1, len(doc)-1)]
        else:
            return False, None
        agg = qdmr.extract_aggregator(" ".join([x.text for x in doc]))
        if not agg:
            return False, None
        meta = [agg]

        return self.update_node(node_id=node_id, graph=graph, params_span=arguments_spans, meta=meta)

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("largest @@1@@", ["AGGREGATE[MAX](@@1@@)"]),
            ("@@1@@ that is largest", ["AGGREGATE[MAX](@@1@@)"]),
        ]


class WrapperFixesBooleanDecomposeRule(OperationDecomposeRule):
    """ Assume to be run before wrapper """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.operation = QDMROperation.BOOLEAN
        self.preserved_tokens = WrapperDecomposeRule._preserved_tokens_map.get(self.operation, [])

    def _decompose(self, node_id: int, graph: nx.DiGraph, doc: [Token]) -> [int]:
        if not doc:
            return False, None

        arguments_spans = []
        meta = []
        if doc[0].lemma_ in ["be", "if"] and all([(x.lemma_ in ["there", "be", "a", "an","any"]) for x in doc[1:-1]]) \
                and isinstance(doc[-1],ReferenceToken):
            arguments_spans = [(len(doc)-1, len(doc)-1)]
            meta = ['EXIST']
        else:
            return False, None

        return self.update_node(node_id=node_id, graph=graph, params_span=arguments_spans, meta=meta)

    def _get_test_cases__str(self) -> (str, [str]):
        return [
            ("if any @@1@@", ["BOOLEAN[EXIST](@@1@@)"]),
            ("if there is a @@1@@", ["BOOLEAN[EXIST](@@1@@)"]),
        ]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    run_tests(OperationDecomposeRule)