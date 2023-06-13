from pathlib import Path

import pytest

from src.graph import CSVGraph, Graph, LocalGraph, Vertex, VertexType


class DummyGraph(Graph):
    def __init__(self) -> None:
        self._vertices: set[Vertex] = set()
        self._edges: set[tuple[Vertex, Vertex, float]] = set()

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


@pytest.fixture(scope="function")
def test_graph() -> DummyGraph:
    return DummyGraph()


@pytest.fixture(scope="function")
def local_graph() -> LocalGraph:
    return LocalGraph()


@pytest.fixture(scope="function")
def csv_graph(root: Path) -> CSVGraph:
    return CSVGraph(root)


ALL_GRAPH_FIXTURES = [
    "test_graph",
    "local_graph",
    "csv_graph",
]


@pytest.mark.parametrize("graph_fixture", ALL_GRAPH_FIXTURES)
def test_add_vertices(request: pytest.FixtureRequest, graph_fixture: str) -> None:
    graph = request.getfixturevalue(graph_fixture)

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


@pytest.mark.parametrize("graph_fixture", ALL_GRAPH_FIXTURES)
def test_add_edges(request: pytest.FixtureRequest, graph_fixture: str) -> None:
    graph = request.getfixturevalue(graph_fixture)

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

    bad_vertex = Vertex("4", VertexType.IMAGE)

    with pytest.raises(KeyError):
        graph.add_edge(bad_vertex, vertex_1)

    with pytest.raises(KeyError):
        graph.add_edge(vertex_1, bad_vertex)


@pytest.mark.parametrize("graph_fixture", ALL_GRAPH_FIXTURES)
def test_add_vertex(request: pytest.FixtureRequest, graph_fixture: str) -> None:
    graph = request.getfixturevalue(graph_fixture)

    assert graph.vertices == set()

    vertex_1 = Vertex("t", VertexType.TAG)
    vertex_2 = Vertex("u", VertexType.USER)
    vertex_3 = Vertex("i", VertexType.IMAGE)

    graph.add_vertex(vertex_1)
    graph.add_vertex(vertex_2)
    graph.add_vertex(vertex_3)

    assert len(graph.vertices) == 3
    assert graph.vertices == {vertex_1, vertex_2, vertex_3}


@pytest.mark.parametrize("graph_fixture", ALL_GRAPH_FIXTURES)
def test_add_directed_edge(request: pytest.FixtureRequest, graph_fixture: str) -> None:
    graph = request.getfixturevalue(graph_fixture)

    vertex_1 = Vertex("t", VertexType.TAG)
    vertex_2 = Vertex("u", VertexType.USER)
    vertex_3 = Vertex("i", VertexType.IMAGE)

    for vertex in [vertex_1, vertex_2, vertex_3]:
        graph.add_vertex(vertex)

        assert graph.out_neighbors(vertex) == dict()
        assert graph.in_neighbors(vertex) == dict()

    graph.add_directed_edge(vertex_1, vertex_2, weight=2.0)

    assert graph.out_neighbors(vertex_1) == {vertex_2: 2.0}
    assert graph.out_neighbors(vertex_2) == dict()
    assert graph.out_neighbors(vertex_3) == dict()
    assert graph.in_neighbors(vertex_1) == dict()
    assert graph.in_neighbors(vertex_2) == {vertex_1: 2.0}
    assert graph.in_neighbors(vertex_3) == dict()
