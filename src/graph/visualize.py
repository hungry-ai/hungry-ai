import networkx as nx  # type: ignore[import]
from pyvis.network import Network  # type: ignore[import]
from .graph import Graph


def visualize(
    graph: Graph,
    file_name: str = "graph_visualization.html",
    weighted: bool = True,
    scaled: bool = True,
) -> None:
    """
    Converts various objects into pyvis network object and outputs visualization as html.
    obj : file name, OR word list, OR word embedding object, OR graph object
    """
    if not file_name.endswith(".html"):
        raise ValueError("file_name must end with .html")

    net = build_net(graph, weighted)
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

    visual_net = Network(notebook=True)
    visual_net.from_nx(net)
    visual_net.show(file_name)
    print("Graph outputted to file: " + file_name)


def build_net(graph: Graph, weighted: bool) -> nx.DiGraph:
    net = nx.DiGraph()
    for vertex, label in graph.labels.items():
        net.add_node(label, group=vertex.type.value)
    for src, dest, weight in graph.edges:
        if weighted:
            net.add_edge(
                graph.labels[src],
                graph.labels[dest],
                title=f"{weight:.2f}",
                arrows="to",
            )
        else:
            net.add_edge(graph.labels[src], graph.labels[dest], arrows="to")
    return net
