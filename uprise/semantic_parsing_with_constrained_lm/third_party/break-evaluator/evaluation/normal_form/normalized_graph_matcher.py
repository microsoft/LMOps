from overrides import overrides
import networkx as nx
from queue import Queue, deque
import logging

import re
import spacy

from  evaluation.decomposition import Decomposition, draw_decomposition_graph
from utils.graph import get_graph_levels

from evaluation.normal_form.normalization_rules import prepare_node
import evaluation.normal_form.normalization_rules as norm_rules
import evaluation.normal_form.operations_normalization_rules as op_norm_rules
from scripts.qdmr_to_program import QDMROperation


_logger = logging.getLogger(__name__)

spacy.prefer_gpu()

class NormalizedGraphMatchScorer:
    def __init__(self, extract_params=True):
        super().__init__()
        self.parser = spacy.load('en_core_web_sm', disable=['ner'])
        self.rules = [
            norm_rules.RemoveDETDecomposeRule(),
            ##op_norm_rules.AggregateDecomposeRule(),
            op_norm_rules.FilterAdjectiveLikeNounDecomposeRule(is_extract_params=extract_params),
            norm_rules.NounsExtractionDecomposeRule(),
            norm_rules.ADPDecomposeRule(),
            norm_rules.CompoundNounExtractionDecomposeRule(),
            norm_rules.AdjectiveDecomposeRule(),
            norm_rules.AdjectiveLikeNounDecomposeRule(),
            op_norm_rules.FilterAdjectiveDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.FilterADPDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.FilterCompoundNounDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.FilterConditionDecomposeRule(is_extract_params=extract_params),
            ##op_norm_rules.SelectionDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.WrapperFixesAggregateDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.WrapperFixesBooleanDecomposeRule(is_extract_params=extract_params),
            op_norm_rules.WrapperDecomposeRule(is_extract_params=extract_params),
        ]
        self._preserved_tokens = [p for r in self.rules for p in r.preserved_tokens]

    def normalized_decomposition(self, decomposition:Decomposition, verbose: bool = False) -> Decomposition:
        norm_g = self.normalize_graph(graph=decomposition.to_graph(), verbose=verbose)
        return Decomposition.from_graph(graph=norm_g)

    def normalize_graph(self, graph: nx.DiGraph, verbose: bool = False) -> nx.DiGraph:
        graph = graph.copy()

        unvisited_nodes = Queue()
        unvisited_nodes.queue = deque(graph.nodes.keys())

        # init: dependencies parsing & POS
        for node_id in unvisited_nodes.queue:
            node = graph.nodes[node_id]
            prepare_node(self.parser, node)

        # decomposition rules
        def run_rules(node_id, is_use_preserved=True):
            is_decomposed = False
            for rule in self.rules:
                preserved_tokens = None
                if is_use_preserved:
                    # todo: preserved words just for no-operational?
                    preserved_tokens = self._preserved_tokens #if (not isinstance(rule, op_norm_rules.OperationDecomposeRule)) else None
                decomposed, added_nodes_ids = rule.decompose(node_id,graph, preserved_tokens=preserved_tokens)
                if decomposed:
                    for id in added_nodes_ids: unvisited_nodes.put(id)
                    is_decomposed = True
                    if verbose:
                        copy = graph.copy()
                        self._update_labels_from_doc(graph=copy)
                        _logger.info(f"{rule}{' -reserved' if is_use_preserved else ''} (node: {node_id})\t{Decomposition.from_graph(graph=copy).to_string()}")
            return is_decomposed

        while not unvisited_nodes.empty():
            node_id = unvisited_nodes.get()
            run_rules(node_id, is_use_preserved=True)
            run_rules(node_id, is_use_preserved=False)

        # update "label" from "doc" if needed
        self._update_labels_from_doc(graph=graph)

        # re-order operation chain
        self.reorder_operations(graph)

        # re-order graph alphabetically
        self.reorder(graph)

        # todo: reorder args: intersection, union, ... args order
        return graph

    @staticmethod
    def _update_labels_from_doc(graph):
        for node in graph.nodes.values():
            if "label" not in node:
                node["label"] = " ".join([t.lemma_ for t in node["doc"]])

    @staticmethod
    def reorder_operations(graph: nx.DiGraph):
        op_to_nodes = {op: {} for op in QDMROperation}
        for node_id, node in graph.nodes.items():
            op = node.get("operation", QDMROperation.NONE)
            op_to_nodes[op][node_id] = node

        # todo: work on a copy of graph (in case of failure)
        NormalizedGraphMatchScorer.reorder_filters_chain(graph, op_to_nodes)

    @staticmethod
    def unwind_refs(graph, field="label"):
        unwind = {}
        _, levels = get_graph_levels(graph)
        for l in sorted(levels.keys()):
            for n_id in levels[l]:
                new = re.sub(r'@@(\d+)@@', lambda x: f"{unwind[int(x.group(1))]}", graph.nodes[n_id][field])
                unwind[n_id] = new
        return unwind

    @staticmethod
    def reorder_filters_chain(graph, op_to_nodes):
        filter_nodes = {k: v for k, v in op_to_nodes[QDMROperation.FILTER].items() if not v.get("meta", [])}

        if not filter_nodes:
            return

        def get_args(node_id):
            node = graph.nodes[node_id]
            arg0, arg1 = re.match(r".*\((.*)\)", node["label"]).group(1).split(",")
            assert norm_rules.ReferenceToken.is_reference(arg0)
            return arg0, arg1

        def head_criteria(node_id):
            pred = list(graph.predecessors(node_id))
            if len(pred) == 1 and pred[0] in filter_nodes:
                arg0, _ = get_args(pred[0])
                return norm_rules.ReferenceToken.get_reference_id(arg0) == node_id
            return False

        head_candidates = {k: v for k, v in filter_nodes.items() if head_criteria(k)}
        visited = {}

        for c_id, c in head_candidates.items():
            try:
                if c_id in visited:
                    continue
                visited[c_id] = None

                # find chain head
                head_id = c_id
                for i in range(len(head_candidates)+1):
                    if i == len(head_candidates):
                        raise ValueError("a cycle was detected")
                    arg0, arg1 = get_args(head_id)
                    arg0_id = norm_rules.ReferenceToken.get_reference_id(arg0)
                    if arg0_id == head_id:
                        raise ValueError("a cycle was detected")
                    if arg0_id not in head_candidates:
                        break
                    head_id = arg0_id

                # find chain items
                chain_items = []
                cur_node_id = head_id
                for i in range(len(filter_nodes)+1):
                    if i == len(filter_nodes) or cur_node_id in chain_items:
                        raise ValueError("a cycle was detected")
                    chain_items.append(cur_node_id)
                    if cur_node_id not in head_candidates:
                        break
                    cur_node_id = list(graph.predecessors(cur_node_id))[0]
                for i in chain_items:
                    visited[i] = None

                # reorder
                unwind = None
                def order_by_arg1(node_id):
                    nonlocal unwind
                    _, arg1 = get_args(node_id)
                    if norm_rules.ReferenceToken.is_reference(arg1):
                        arg1_id = norm_rules.ReferenceToken.get_reference_id(arg1)
                        unwind = unwind or NormalizedGraphMatchScorer.unwind_refs(graph)
                        return unwind[arg1_id]
                        #return graph.nodes[arg1_id]["label"]
                    return re.sub(r"@@(\d+)@@", "", arg1).replace("  ", " ")

                sorted_chain = sorted(chain_items, key=lambda x: order_by_arg1(x))
                head_arg0, head_arg1 = get_args(chain_items[0])
                chain_successor = norm_rules.ReferenceToken.get_reference_id(head_arg0)
                chain_predecessors = list(graph.predecessors(chain_items[-1]))

                # remove current edges
                graph.remove_edge(chain_items[0], chain_successor)
                for i in range(len(chain_items)-1):
                    graph.remove_edge(chain_items[i+1], chain_items[i])
                for p in chain_predecessors:
                    graph.remove_edge(p, chain_items[-1])

                # add new edges
                def update_arg0(node_id, value):
                    node = graph.nodes[node_id]
                    arg0, _ = get_args(node_id)
                    node["label"] = node["label"].replace(arg0, f"@@{value}@@")
                    graph.add_edge(node_id, value)

                update_arg0(sorted_chain[0], chain_successor)
                for i in range(len(sorted_chain)-1):
                    update_arg0(sorted_chain[i+1], sorted_chain[i])
                for p in chain_predecessors:
                    graph.add_edge(p, sorted_chain[-1])
                    p_node = graph.nodes[p]
                    p_node["label"] = p_node["label"].replace(f"@@{chain_items[-1]}@@", f"@@{sorted_chain[-1]}@@")
            except ValueError as ex:
                _logger.warning("skip - reorder head in reorder filter chain")


    @staticmethod
    def reorder(graph: nx.DiGraph):
        try:
            _, levels = get_graph_levels(graph)
        except Exception as ex:
            _logger.warning("skip - Failed to invoke reorder()")
            return

        # order by levels
        relabel_map = {}
        next_node_id = 1
        for level in sorted(levels.keys()):
            nodes_by_label = {}
            for n_id in levels[level]:
                # for consistency
                n = graph.nodes[n_id]
                n["label"] = re.sub(r"@@(\d+)@@", lambda x: f"@@{relabel_map[int(x.group(1))]}@@", n["label"])
                if n["label"] not in nodes_by_label:
                    nodes_by_label[n["label"]] = n_id
                else:
                    # duplicate label
                    # safe merge since previous levels are fixed
                    exists_node_id = nodes_by_label[n["label"]]
                    # successors - should be the same due to label
                    predecessors = graph.predecessors(n_id)
                    for p_id in predecessors:
                        graph.add_edge(p_id, exists_node_id)
                        graph.nodes[p_id]["label"] = re.sub(f"@@{n_id}@@", lambda x: f"@@{exists_node_id}@@", graph.nodes[p_id]["label"])
                    graph.remove_node(n_id)
            nodes_order = sorted(nodes_by_label.keys())
            for label in nodes_order:
                n_id = nodes_by_label[label]
                relabel_map[n_id]=next_node_id
                next_node_id += 1

        # update labels
        # double mapping since new and old labels are overlap
        nx.relabel.relabel_nodes(graph, {k:str(v) for k,v in relabel_map.items()}, copy=False)
        nx.relabel.relabel_nodes(graph, {str(v):v for v in relabel_map.values()}, copy=False)
        # note: node "labels" are updated


def test_exact_match(html:str, dest=None, verbose = False):
    import pandas as pd
    import traceback
    from model.model_base import ModelBase
    import time
    start_time = time.time()

    df = pd.read_html(html)[0]
    norm_g = NormalizedGraphMatchScorer()

    def predictions_to_norm(df, qdmr_col, norm_col):
        df[norm_col] = df[qdmr_col]

        for index, row in df.iterrows():
            dec = row[norm_col]
            try:
                decomposition = norm_g.normalized_decomposition(Decomposition.from_str(dec))
                df.loc[index, norm_col] = decomposition.to_string()
            except Exception as ex:
                print(f"error in index {index}:{str(ex)}\n{dec}", flush=True)
                traceback.print_exc()
                df.loc[index, norm_col] = "ERROR"

    norm_rules.load_cache(html.replace(".html", "__cache"))
    predictions_to_norm(df, "gold", "gold_norm")
    predictions_to_norm(df, "prediction", "prediction_norm")
    norm_rules.save_cache(html.replace(".html", "__cache"))

    df["raw exact match"] = df["gold"].str.lower() == df["prediction"].str.lower()
    df["norm exact match"] = (df["gold_norm"] == df["prediction_norm"]) & (df["gold_norm"] != 'ERROR') & (
                df["prediction_norm"] != 'ERROR')

    print(html, flush=True)
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    print(f"rules:{norm_g.rules}")
    diff = df[df["raw exact match"]!=df["norm exact match"]]
    if verbose:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(diff[["raw exact match", "norm exact match", "gold","prediction","gold_norm", "prediction_norm"]])

    ModelBase.print_score_stats({
        "raw exact match": df["raw exact match"].tolist(),
        "norm exact match": df["norm exact match"].tolist()
    })

    if dest is not None:
        df.to_html(dest)
        diff.to_html(dest.replace(".html", "__diff.html"))

    regression = df[(df["raw exact match"] == True) & (df["norm exact match"] == False)]
    if len(regression.index) > 0:
        message = f"regression: {len(regression.index)}"
        print(message, flush=True)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(regression[["raw exact match", "norm exact match", "gold","prediction","gold_norm", "prediction_norm"]])
        raise Exception(message)
    return df


def compare_performances(df1, df2, suffixes:(str,str)=(' (1)',' (2)')):
    import pandas as pd
    s1, s2 = suffixes
    assert s1 != s2
    cols = ["question", "gold", "prediction","gold_norm", "prediction_norm", "raw exact match", "norm exact match"]
    df1 = df1[cols].rename(columns={c:c+s1 for c in cols})
    df2 = df2[cols[3:]].rename(columns={c:c+s2 for c in cols})
    df = pd.concat([df1, df2], axis=1, sort=False)
    diff = df[df["norm exact match"+s1]!=df["norm exact match"+s2]]
    regression_amount = len(diff[(diff["norm exact match"+s1]==True) & (diff["norm exact match"+s2]==False)].index)
    return diff, regression_amount


def compare_to(df, norm_html:str):
    import pandas as pd
    import os
    df2 = pd.read_html(norm_html)[0]
    diff, reg = compare_performances(df2, df, suffixes=('', ' (new)'))
    diff.to_html(os.path.join(os.path.dirname(norm_html), f"compared_diff__{os.path.basename(norm_html)}"))
    _logger.info(f"regression of: {reg}")



def plot_normal_decomposition(decomp: str):
    normalizer = NormalizedGraphMatchScorer()
    decomposition = Decomposition.from_str(decomp)
    print("decomposition:", decomposition.to_string())
    norm_decomposition = normalizer.normalized_decomposition(decomposition, verbose=True)
    print("normal form:", norm_decomposition.to_string())
    print("=========================================================")
    draw_decomposition_graph(decomposition.to_graph(), title="decomposition")
    draw_decomposition_graph(norm_decomposition.to_graph(), title="normal form")


def test(save: bool = False, verbose: bool = False):
    save_path_random = "/media/disk1/matan/questions_decomp/_debug/qdecomp/tmp/seq2seq-copynet--bert_layers2_lr0.001_hd450_dop0.2_low_level__random_norm.html" if save else None
    df_random = test_exact_match("/media/disk1/matan/questions_decomp/qdecomp/tmp/g02/tune/data_old_version/seq2seq-copynet--bert_low_level/seq2seq-copynet--bert_layers2_lr0.001_hd450_dop0.2_low_level/evals/dev_low_level_evals_full_samples/html/seq2seq-copynet--bert_layers2_lr0.001_hd450_dop0.2_low_level__random.html",
                                 save_path_random,
                                 verbose=verbose)
    compare_to(df_random, "/media/disk1/matan/questions_decomp/_debug/qdecomp/tmp/seq2seq-copynet--bert_layers2_lr0.001_hd450_dop0.2_low_level__random_norm.html")

    save_path_dev = "/media/disk1/matan/questions_decomp/_debug/qdecomp/tmp/seq2seq-copynet--bert_layers2_lr0.001_hd450_dop0.2_low_level__all_norm.html" if save else None
    df_dev = test_exact_match("/media/disk1/matan/questions_decomp/qdecomp/tmp/g02/tune/data_old_version/seq2seq-copynet--bert_low_level/seq2seq-copynet--bert_layers2_lr0.001_hd450_dop0.2_low_level/evals/dev_low_level_evals_full_samples/html/seq2seq-copynet--bert_layers2_lr0.001_hd450_dop0.2_low_level__all.html",
                              save_path_dev
                              )
    compare_to(df_dev, "/media/disk1/matan/questions_decomp/_debug/qdecomp/tmp/seq2seq-copynet--bert_layers2_lr0.001_hd450_dop0.2_low_level__all_norm.html")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    test()
    #plot_normal_decomposition("")
