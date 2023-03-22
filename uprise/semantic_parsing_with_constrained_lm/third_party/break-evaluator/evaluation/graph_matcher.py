
import heapq

import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np

from itertools import chain, combinations, permutations
from multiprocessing import Pool
from progressbar import ProgressBar, SimpleProgress
from tqdm import tqdm

from evaluation.sequence_matcher import SequenceMatchScorer
from utils.timeout import exit_after


class GraphMatchScorer(object):
    def __init__(self):
        self.sequence_matcher = SequenceMatchScorer(remove_stop_words=False)

    def node_subst_cost_lexical(self, node1, node2):
        return 1 - self.sequence_matcher.get_match_score(node1['label'], node2['label'])

    ###@exit_after(600) 
    @exit_after(180) ### quit after 3 minutes
    def normalized_graph_edit_distance(self, graph1, graph2, structure_only):
        """Returns graph edit distance normalized between [0,1].

        Parameters
        ----------
        graph1 : graph
        graph2 : graph
        structure_only : whether to use node substitution cost 0 (e.g. all nodes are identical).

        Returns
        -------
        float
            The normalized graph edit distance of G1,G2.
            Node substitution cost is normalized string edit distance of labels.
            Insertions cost 1, deletion costs 1.
        """
        if structure_only:
            node_subst_cost = lambda x, y: 0
        else:
            node_subst_cost = self.node_subst_cost_lexical
        approximated_distances = nx.optimize_graph_edit_distance(graph1, graph2,
                                                                 node_subst_cost=node_subst_cost)
        total_cost_graph1 = len(graph1.nodes) + len(graph1.edges)
        total_cost_graph2 = len(graph2.nodes) + len(graph2.edges)
        normalization_factor = max(total_cost_graph1, total_cost_graph2)

        dist = None
        for v in approximated_distances:
            dist = v

        return float(dist)/normalization_factor

    def get_edit_distance_match_scores(self, predictions, targets, structure_only=False):
        distances = []
        num_examples = len(predictions)
        for i in tqdm(range(num_examples)):
            try:
                dist = self.normalized_graph_edit_distance(predictions[i], targets[i],
                                                           structure_only)
            except KeyboardInterrupt:
                print(f"skipping example: {i}")
                dist = None

            distances.append(dist)

        return distances


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)

    def count(self):
        return len(self.elements)


class AStarSearcher(object):
    def __init__(self):
        self.sequence_matcher = SequenceMatchScorer(remove_stop_words=False)
        self.graph1 = None
        self.graph2 = None

    def _get_label(self, graph_number, node):
        assert graph_number in [1, 2]
        if graph_number == 1:
            return self.graph1.nodes[node]['label']
        else:
            return self.graph2.nodes[node]['label']

    def _get_edit_ops(self, edit_path):
        edit_ops = set()
        graph1_matched_nodes, graph2_matched_nodes = self._get_matched_nodes(edit_path)
        graph1_unmatched_nodes = [node for node in self.graph1.nodes if node not in graph1_matched_nodes]
        graph2_unmatched_nodes = [node for node in self.graph2.nodes if node not in graph2_matched_nodes]

        if not edit_path:
            i = 1
        else:
            i = min(graph1_unmatched_nodes)

        subsets1 = self._get_all_subsets(graph1_unmatched_nodes)
        subsets2 = self._get_all_subsets(graph2_unmatched_nodes)

        # for i in graph1_unmatched_nodes:
        # add {v_i -> u_j}, {v_i -> u_j+u_j+1}, ...
        for subset in subsets2:
            edit_ops.add(((i,), subset))

        # add {v_i -> del}
        edit_ops.add(((i,), (-1,)))

        # add {v_i+v_i+1 -> u_j}, ...
        for subset in subsets1:
            if i in subset:
                for j in graph2_unmatched_nodes:
                    edit_ops.add((subset, (j,)))

        return list(edit_ops)

    @staticmethod
    def _get_edge_crossing_cost(edit_path):
        segments = []
        for edit_op in edit_path:
            for source in edit_op[0]:
                for target in edit_op[1]:
                    segments.append((source, target))

        cost = 0.0
        for i in range(len(segments)):
            for j in range(i+1, len(segments)):
                # all segments have the same orientation (going down from source to target).
                if (segments[i][0] < segments[j][0] and segments[i][1] > segments[j][1]) or \
                        (segments[i][0] > segments[j][0] and segments[i][1] < segments[j][1]):
                    cost += 1.0

        return cost

    def _get_merge_op_cost(self, merge_nodes, target):
        graph2_label = self._get_label(2, target)
        min_merge_cost = 1
        for permutation in permutations(merge_nodes):
            graph1_labels = [self._get_label(1, node) for node in permutation]
            graph1_label = ' '.join(graph1_labels)
            permutation_merge_cost = 1 - self.sequence_matcher.get_match_score(graph1_label, graph2_label)

            if permutation_merge_cost < min_merge_cost:
                min_merge_cost = permutation_merge_cost

        return min_merge_cost

    def _get_split_op_cost(self, source, split_nodes):
        graph1_label = self._get_label(1, source)
        min_split_cost = 1
        for permutation in permutations(split_nodes):
            graph2_labels = [self._get_label(2, node) for node in permutation]
            graph2_label = ' '.join(graph2_labels)
            permutation_split_cost = 1 - self.sequence_matcher.get_match_score(graph1_label, graph2_label)

            if permutation_split_cost < min_split_cost:
                min_split_cost = permutation_split_cost

        return min_split_cost

    def _get_edit_path_cost(self, edit_path):
        cost = 0

        for edit_op in edit_path:
            # node insertion
            if edit_op[0] == (-1,):
                cost += 1

            # node deletion
            elif edit_op[1] == (-1,):
                cost += 1

            # node substitution
            elif len(edit_op[0]) == len(edit_op[1]):
                graph1_label = self._get_label(1, edit_op[0][0])
                graph2_label = self._get_label(2, edit_op[1][0])
                substitution_cost = 1 - self.sequence_matcher.get_match_score(graph1_label, graph2_label)
                cost += substitution_cost

            # node merging
            elif len(edit_op[0]) > 1:
                min_merge_cost = self._get_merge_op_cost(edit_op[0], edit_op[1][0])
                cost += min_merge_cost * len(edit_op[0])

            # node splitting
            elif len(edit_op[1]) > 1:
                min_split_cost = self._get_split_op_cost(edit_op[0][0], edit_op[1])
                cost += min_split_cost * len(edit_op[1])

            else:
                raise RuntimeError(
                    "get_edit_op_cost: edit op does not match any edit type: {}".format(edit_op)
                )

        edge_crossing_cost = self._get_edge_crossing_cost(edit_path)

        return cost + edge_crossing_cost

    def _get_heuristic_cost(self, edit_path):
        # graph1_curr_nodes, graph2_curr_nodes = self._get_matched_nodes(edit_path)
        # heuristic_cost = num_graph1_unmatched_nodes + num_graph2_unmatched_nodes
        # num_graph1_unmatched_nodes = graph1.number_of_nodes() - len(graph1_curr_nodes)
        # num_graph2_unmatched_nodes = graph2.number_of_nodes() - len(graph2_curr_nodes)

        return 0

    def _get_curr_edit_path_string(self, edit_path):
        result = []

        for edit_op in edit_path:
            source = ','.join([self._get_label(1, node) if node != -1 else '-' for node in edit_op[0]])
            target = ','.join([self._get_label(2, node) if node != -1 else '-' for node in edit_op[1]])
            result.append('[{}]->[{}]'.format(source, target))

        return ', '.join(result)

    def _is_isomorphic_graphs(self):
        nm = iso.categorical_node_match('label', '')
        em = iso.numerical_edge_match('weight', 1)
        return nx.is_isomorphic(self.graph1, self.graph2,
                                node_match=nm, edge_match=em)

    @staticmethod
    def get_edit_op_count(edit_path):
        edit_op_counts = {
            "insertion": 0,
            "deletion": 0,
            "substitution": 0,
            "merging": 0,
            "splitting": 0
        }

        for edit_op in edit_path:
            if edit_op[0] == (-1,):
                edit_op_counts["insertion"] += 1

            elif edit_op[1] == (-1,):
                edit_op_counts["deletion"] += 1

            elif len(edit_op[1]) > 1:
                edit_op_counts["splitting"] += 1

            elif len(edit_op[0]) == len(edit_op[1]) == 1:
                edit_op_counts["substitution"] += 1

            elif len(edit_op[0]) > 1:
                edit_op_counts["merging"] += 1

            else:
                raise RuntimeError("_get_edit_op_type: edit op type was not identified: {}".format(edit_op))

        return edit_op_counts

    @staticmethod
    def _get_all_subsets(ss):
        subsets = chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))
        subsets = [subset for subset in subsets if len(subset) > 0]

        return subsets

    @staticmethod
    def _get_matched_nodes(edit_path):
        graph1_matched_nodes, graph2_matched_nodes = [], []
        for (graph1_nodes, graph2_nodes) in edit_path:
            graph1_matched_nodes.extend([node for node in graph1_nodes if node != -1])
            graph2_matched_nodes.extend([node for node in graph2_nodes if node != -1])

        return graph1_matched_nodes, graph2_matched_nodes

    def set_graphs(self, graph1, graph2):
        self.graph1 = graph1
        self.graph2 = graph2

    def a_star_search(self, debug=False):
        assert self.graph1 and self.graph2

        if self._is_isomorphic_graphs():
            self.graph1, self.graph2 = None, None
            return [], "", 0, 0, 0, 0

        found_best_path = False
        queue = PriorityQueue()
        edit_ops = self._get_edit_ops([])
        for edit_op in edit_ops:
            queue.put([edit_op],
                      self._get_edit_path_cost([edit_op]) +
                      self._get_heuristic_cost([edit_op]))
        num_ops = len(edit_ops)

        while True:
            if queue.empty():
                raise RuntimeError("a_star_search: could not find a complete edit path.")

            curr_cost, curr_edit_path = queue.get()
            graph1_curr_nodes, graph2_curr_nodes = self._get_matched_nodes(curr_edit_path)

            if len(graph1_curr_nodes) < self.graph1.number_of_nodes():
                edit_ops = self._get_edit_ops(curr_edit_path)
                for edit_op in edit_ops:
                    curr_edit_path_extended = curr_edit_path + [edit_op]
                    queue.put(curr_edit_path_extended,
                              self._get_edit_path_cost(curr_edit_path_extended) +
                              self._get_heuristic_cost(curr_edit_path_extended))
                num_ops += len(edit_ops)

            elif len(graph2_curr_nodes) < self.graph2.number_of_nodes():
                edit_ops = [((-1,), (node,)) for node in self.graph2.nodes
                            if node not in graph2_curr_nodes]
                curr_edit_path_extended = curr_edit_path + edit_ops
                queue.put(curr_edit_path_extended,
                          self._get_edit_path_cost(curr_edit_path_extended) +
                          self._get_heuristic_cost(curr_edit_path_extended))
                num_ops += len(edit_ops)

            elif debug:
                if not found_best_path:
                    found_best_path = True
                    best_edit_path, best_cost = curr_edit_path, curr_cost
                    num_paths_true = queue.count() + 1
                    num_ops_true = num_ops
                    num_ops = 0
                elif not queue.empty():
                    continue
                else:
                    best_edit_path_string = self._get_curr_edit_path_string(best_edit_path)
                    explored_ops_ratio = (num_ops_true * 100.0) / (num_ops_true + num_ops)
                    # print("explored {:.2f}% of the ops.".format(explored_ops_ratio))
                    self.graph1, self.graph2 = None, None
                    return best_edit_path, best_edit_path_string, best_cost, num_paths_true, num_ops_true,\
                           explored_ops_ratio

            else:
                curr_edit_path_string = self._get_curr_edit_path_string(curr_edit_path)
                num_paths = queue.count() + 1   # +1 for the current path we just popped

                self.graph1, self.graph2 = None, None
                return curr_edit_path, curr_edit_path_string, curr_cost, num_paths, num_ops, None


def get_ged_plus_score(idx, graph1, graph2, exclude_thr, debug):
    if exclude_thr and \
            (graph1.number_of_nodes() > exclude_thr or
             graph2.number_of_nodes() > exclude_thr):
        print(f"skipping example: {idx}")
        return idx, None, None, None, None, None, None

    a_start_searcher = AStarSearcher()
    a_start_searcher.set_graphs(graph1, graph2)
    curr_edit_path, _, curr_cost, curr_num_paths, curr_num_ops, curr_ops_ratio = \
        a_start_searcher.a_star_search(debug=debug)
    curr_edit_op_counts = a_start_searcher.get_edit_op_count(curr_edit_path)

    return idx, curr_edit_path, curr_cost, curr_num_paths, curr_num_ops,\
           curr_ops_ratio, curr_edit_op_counts


def get_ged_plus_scores(decomposition_graphs, gold_graphs,
                        exclude_thr=None, debug=False, num_processes=5):
    samples = list(zip(decomposition_graphs, gold_graphs))
    pool = Pool(num_processes)
    pbar = ProgressBar(widgets=[SimpleProgress()], maxval=len(samples)).start()

    results = []
    _ = [pool.apply_async(get_ged_plus_score,
                          args=(i, samples[i][0], samples[i][1], exclude_thr, debug),
                          callback=results.append)
         for i in range(len(samples))]
    while len(results) < len(samples):
        pbar.update(len(results))
    pbar.finish()
    pool.close()
    pool.join()

    edit_op_counts = {
        "insertion": 0,
        "deletion": 0,
        "substitution": 0,
        "merging": 0,
        "splitting": 0
    }
    idxs, scores_tmp, num_paths, num_ops, ops_ratio = [], [], [], [], []
    for result in results:
        idx, curr_edit_path, curr_cost, curr_num_paths, curr_num_ops, curr_ops_ratio, curr_edit_op_counts = result
        idxs.append(idx)
        scores_tmp.append(curr_cost)
        if not curr_cost:
            continue

        num_paths.append(float(curr_num_paths))
        num_ops.append(float(curr_num_ops))
        if debug:
            ops_ratio.append(curr_ops_ratio)

        for op in curr_edit_op_counts:
            edit_op_counts[op] += curr_edit_op_counts[op]

    scores = [score for (idx, score) in sorted(zip(idxs, scores_tmp))]

    print("edit op statistics:", edit_op_counts)
    print("number of explored paths: mean {:.2}, min {:.2}, max {:.2}".format(
        np.mean(num_paths), np.min(num_paths), np.max(num_paths)))
    print("number of edit ops: mean {:.2}, min {:.2}, max {:.2}".format(
        np.mean(num_ops), np.min(num_ops), np.max(num_ops)))
    if debug:
        print("explored ops ratio: mean {:.2}, min {:.2}, max {:.2}".format(
            np.mean(ops_ratio), np.min(ops_ratio), np.max(ops_ratio)))

    return scores

