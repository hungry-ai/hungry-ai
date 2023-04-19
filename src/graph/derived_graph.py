from collections import defaultdict

from .graph import Graph
from .vertex import Vertex, VertexType


class DerivedGraph(Graph):
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def add_vertex(self, vertex: Vertex) -> None:
        raise ValueError("derived graphs are immutable")

    @property
    def vertices(self) -> set[Vertex]:
        return self.graph.vertices

    def add_directed_edge(self, src: Vertex, dest: Vertex, weight: float = 1.0) -> None:
        raise ValueError("derived graphs are immutable")

    def out_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        if vertex.type in (VertexType.IMAGE, VertexType.TAG):
            return self.graph.out_neighbors(vertex)

        out_neighbors = self.graph.out_neighbors(vertex)
        if any(neighbor.type != VertexType.IMAGE for neighbor in out_neighbors):
            raise ValueError("improperly formed graph")

        second_neighbors = defaultdict(list)
        for x1, w1 in out_neighbors(vertex):
            for x2, w2 in self.graph.out_neighbors(x1).items():
                second_neighbors[x2].append((w1, w2))

        return {
            x: sum(w1 * w2 for w1, w2 in weights) / sum(w2 for _, w2 in weights)
            for x, weights in second_neighbors.items()
        }

    def in_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        in_neighbors = self.graph.in_neighbors(vertex)

        second_neighbors = {
            x2: self.out_neighbors(x2)[vertex]
            for x1 in in_neighbors
            if x1.type == VertexType.IMAGE
            for x2 in self.graph.in_neighbors(x1)
            if x2.type == VertexType.USER
        }

        return {**in_neighbors, **second_neighbors}
