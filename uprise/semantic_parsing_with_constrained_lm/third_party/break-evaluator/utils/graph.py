import networkx as nx
from queue import Queue, deque


def has_cycle(graph: nx.DiGraph):
    try:
        nx.find_cycle(graph, orientation='original')
        return True
    except:
        return False


def get_graph_levels(graph: nx.DiGraph):
    """
    Find graph level for each node
    level[node] := 0 if the node has no successors
    level[node] := max[over successors s](level[s])+1
    :param graph: directed graph with no cycles
    :return: (nodes_level, levels) tuple where:
        nodes_level: dictionary of <node_id>:<level:int>
        levels: dictionary of <level:int>:[<node_id>]
    """
    updated_nodes = Queue()

    # first layer
    leafs = [n_id for n_id in graph.nodes if not any(graph.successors(n_id))]
    nodes_levels = {n_id: 0 for n_id in leafs}
    updated_nodes.queue = deque(leafs)

    # update predecessors
    while not updated_nodes.empty():
        n_id = updated_nodes.get()
        low_bound = nodes_levels[n_id] + 1
        if low_bound > graph.number_of_nodes():
            raise ValueError("Cyclic graphs are not allowed")
        for s_id in graph.predecessors(n_id):
            if nodes_levels.get(s_id, -1) < low_bound:
                nodes_levels[s_id] = low_bound
                updated_nodes.put(s_id)
    levels = {}
    for n_id, l in nodes_levels.items():
        levels[l] = levels.get(l, []) + [n_id]

    return nodes_levels, levels
