from src.graph import Graph, Vertex, VertexType


class TestGraph(Graph):
    def __init__(self) -> None:
        self._vertices = set()
        self._edges = set()

    def add_vertex(self, vertex: Vertex) -> None:
        self._vertices.add(vertex)

    def add_directed_edge(self, src: Vertex, dest: Vertex, weight: float = 1.0) -> None:
        self._edges.add((src, dest, weight))

    @property
    def vertices(self) -> set[Vertex]:
        return self._vertices

    def out_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        return {dest: weight for src, dest, weight in self._edges if src == vertex}

    def in_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        return {src: weight for src, dest, weight in self._edges if dest == vertex}


def test_add_vertices() -> None:
    graph = TestGraph()

    vertex_1 = graph.add_tag("1")
    assert vertex_1.type == VertexType.TAG

    vertex_2 = graph.add_user("2")
    assert vertex_2.type == VertexType.USER

    vertex_3 = graph.add_image("3")
    assert vertex_3.type == VertexType.IMAGE

    assert graph.vertices == {vertex_1, vertex_2, vertex_3}
    assert graph.tags == {vertex_1}
    assert graph.users == {vertex_2}
    assert graph.images == {vertex_3}

    graph.add_tag("1")
    graph.add_user("1")
    graph.add_image("1")
    graph.add_tag("2")
    graph.add_user("2")
    graph.add_image("2")
    graph.add_tag("3")
    graph.add_user("3")
    graph.add_image("3")

    assert len(graph.vertices) == 9


def test_add_edges() -> None:
    graph = TestGraph()

    vertex_1 = graph.add_tag("1")
    vertex_2 = graph.add_tag("2")
    vertex_3 = graph.add_tag("3")

    graph.add_edge(vertex_1, vertex_2, weight=2.0)
    graph.add_edge(vertex_2, vertex_3, directed=False)

    assert graph.out_neighbors(vertex_1) == {vertex_2: 2.0}
    assert graph.out_neighbors(vertex_2) == {vertex_3: 1.0}
    assert graph.out_neighbors(vertex_3) == {vertex_2: 1.0}
    assert graph.in_neighbors(vertex_1) == dict()
    assert graph.in_neighbors(vertex_2) == {vertex_1: 2.0, vertex_3: 1.0}
    assert graph.in_neighbors(vertex_3) == {vertex_2: 1.0}
