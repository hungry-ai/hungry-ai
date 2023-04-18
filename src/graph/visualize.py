import networkx as nx  # type: ignore[import]
from pyvis.network import Network  # type: ignore[import]
from typing import Any

from .graph import Graph, Vertex


def visualize(
    graph: Graph,
    labels: dict[Vertex, str],
    file_name: str = "visualize.html",
    weighted: bool = True,
    scaled: bool = True,
) -> Any:
    """
    Converts graph into pyvis network object and outputs visualization as html.
    graph: Graph object
    """
    if not file_name.endswith(".html"):
        raise ValueError("file_name must end with .html")

    net = build_net(graph, labels, weighted)
    if scaled:
        in_dict = dict(net.in_degree)
        min_size = 5
        scale = 10
        in_dict.update(
            (word, min_size + scale * in_degree) for word, in_degree in in_dict.items()
        )
        nx.set_node_attributes(net, in_dict, "size")
    else:
        nx.set_node_attributes(net, "circle", "shape")

    visual_net = Network(notebook=True, cdn_resources="in_line")
    visual_net.from_nx(net)
    print("Graph outputted to file: " + str(file_name))
    return visual_net.show(file_name)


def build_net(graph: Graph, labels: dict[Vertex, str], weighted: bool) -> nx.DiGraph:
    net = nx.DiGraph()
    for vertex, label in labels.items():
        net.add_node(label, group=vertex.type.value)
    for src in graph.vertices:
        for dest, weight in graph.out_neighbors(src).items():
            if weighted:
                net.add_edge(
                    labels[src],
                    labels[dest],
                    title=f"{weight:.2f}",
                    arrows="to",
                )
            else:
                net.add_edge(labels[src], labels[dest], arrows="to")
    return net
