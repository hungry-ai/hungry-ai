from typing import Any

import networkx as nx  # type: ignore[import]
from pyvis.network import Network  # type: ignore[import]

from .graph import Graph, Vertex


def visualize(
    graph: Graph,
    labels: dict[Vertex, str] = {},
    file_name: str = "visualize.html",
    weighted: bool = True,
    scaled: bool = True,
    path: set[Vertex] = None
) -> Any:
    """
    Converts graph into pyvis network object and outputs visualization as html.
    graph: Graph object
    """
    if not str(file_name).endswith(".html"):
        raise ValueError("file_name must end with .html")
    if labels is None:
        raise TypeError("Please pass a labels dict.")

    net = build_net(graph, labels, weighted, path)
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

    visual_net = Network(notebook=True)#, cdn_resources="in_line")
    visual_net.from_nx(net)
    print("Graph outputted to file: " + str(file_name))
    return visual_net.show(str(file_name))

def build_net(
    graph: Graph, labels: dict[Vertex, str], weighted: bool, path: set[Vertex]
) -> nx.DiGraph:
    net = nx.DiGraph()
    for vertex, label in labels.items():
        if path and vertex in path:
            net.add_node(label, group=4)
        else:
            net.add_node(label, group=vertex.type.value)
    for src in graph.vertices:
        for dest, weight in graph.out_neighbors(src).items():
            if weighted:
                net.add_edge(
                    labels.get(src, src.name),
                    labels.get(dest, dest.name),
                    title=f"{weight:.2f}",
                    arrows="to",
                )
            else:
                net.add_edge(
                    labels.get(src, src.name), labels.get(dest, dest.name), arrows="to"
                )
    return net

def build_path(parent: dict[Vertex, Vertex]) -> set[Vertex]:
    path = set()
    for src, dest in parent.items():
        path.add(src)
        path.add(dest)
    return path
