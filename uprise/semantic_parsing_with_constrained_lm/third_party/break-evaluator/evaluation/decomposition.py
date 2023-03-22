
import matplotlib.pyplot as plt
import networkx as nx
import re

from utils.graph import get_graph_levels

class Decomposition(object):
    def __init__(self, decomposition_list):
        self.decomposition_list = [str(step) for step in decomposition_list]

    def _get_graph_edges(self):
        edges = []
        for i, step in enumerate(self.decomposition_list):
            references = self._get_references_ids(step)
            step_edges = [(i+1, ref) for ref in references]
            edges.extend(step_edges)

        return edges

    def to_string(self):
        return " @@SEP@@ ".join([x.replace("  ", " ").strip() for x in self.decomposition_list])

    @staticmethod
    def _get_references_ids(step):
        return [int(x) for x in re.findall(r"@@(\d+)@@", step)]

    @staticmethod
    def from_str(text, sep="@@SEP@@"):
        decomposition_list = [re.sub(r"\s+", " ", x.strip()) for x in re.split(sep, text, flags=re.IGNORECASE)]
        return Decomposition(decomposition_list)

    @staticmethod
    def from_tokens(decomposition_tokens, sep="@@SEP@@"):
        decomposition_str = ' '.join(decomposition_tokens)
        return Decomposition.from_str(decomposition_str, sep)

    def to_graph(self, nodes_only=False):
        # initiate a directed graph
        graph = nx.DiGraph()

        # add edges
        if nodes_only:
            edges = []
        else:
            edges = self._get_graph_edges()
        graph.add_edges_from(edges)

        # add nodes
        nodes = self.decomposition_list
        for i in range(len(nodes)):
            graph.add_node(i+1, label=nodes[i])

        # handle edge cases where artificial nodes need to be added
        for node in graph.nodes:
            if 'label' not in graph.nodes[node]:
                graph.add_node(node, label='')

        return graph

    @staticmethod
    def from_graph(graph: nx.DiGraph):
        decomposition_list = [graph.nodes[i+1]["label"] for i in range(graph.number_of_nodes())]
        return Decomposition(decomposition_list)

    def draw_decomposition(self):
        graph = self.to_graph(False)
        draw_decomposition_graph(graph)


def draw_decomposition_graph(graph, title=None, pos=None):
    options = {
        'node_color': 'lightblue',
        'node_size': 400,
        'width': 1,
        'arrowstyle': '-|>',
        'arrowsize': 14,
    }

    if not pos:
        try:
            _, levels = get_graph_levels(graph)

            pos = {}
            max_num_nodes_per_layer = max([len(levels[l]) for l in levels]) + 1
            for l in levels.keys():
                num_layer_nodes = len(levels[l])
                if num_layer_nodes == 0: continue
                space_factor = max_num_nodes_per_layer // (num_layer_nodes + 1)
                for i, n_id in enumerate(sorted(levels[l])):
                    pos[n_id] = ((i + 1) * space_factor, l)
        except:  # cyclic
            print("a cyclic graph - change layout (no-layers)")
            pos = nx.spring_layout(graph, k=0.5)
    nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, **options)
    for node in graph.nodes:
        plt.text(pos[node][0], pos[node][1]-0.1,
                 s=re.sub(r'@@(\d+)@@', r'#\g<1>', graph.nodes[node]['label']),
                 bbox=dict(facecolor='red', alpha=0.5),
                 horizontalalignment='center',
                 wrap=True, size=6)

    if title:
        plt.title(title)

    plt.axis("off")
    plt.show()
    plt.clf()

