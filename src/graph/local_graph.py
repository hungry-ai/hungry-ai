from .graph import Graph
from .vertex import Vertex


class LocalGraph(Graph):
    def __init__(self) -> None:
        self._vertices: set[Vertex] = set()

        self._out_neighbors: dict[Vertex, dict[Vertex, float]] = dict()
        self._in_neighbors: dict[Vertex, dict[Vertex, float]] = dict()

    def add_vertex(self, vertex: Vertex) -> None:
        if vertex in self._vertices:
            pass

        self._vertices.add(vertex)
        self._out_neighbors[vertex] = dict()
        self._in_neighbors[vertex] = dict()

    @property
    def vertices(self) -> set[Vertex]:
        return self._vertices

    def add_directed_edge(self, src: Vertex, dest: Vertex, weight: float = 1.0) -> None:
        self._out_neighbors[src][dest] = weight
        self._in_neighbors[dest][src] = weight

    def out_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        return self._out_neighbors[vertex]

    def in_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        return self._in_neighbors[vertex]
