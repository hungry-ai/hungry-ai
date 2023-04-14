import pytest

from src.graph import LocalGraph, Vertex, VertexType


def test_add_vertex() -> None:
    graph = LocalGraph()

    assert graph.vertices == set()

    vertex_1 = Vertex("t", VertexType.TAG)
    vertex_2 = Vertex("u", VertexType.USER)
    vertex_3 = Vertex("i", VertexType.IMAGE)

    graph.add_vertex(vertex_1)
    graph.add_vertex(vertex_2)
    graph.add_vertex(vertex_3)

    assert len(graph.vertices) == 3
    assert graph.vertices == {vertex_1, vertex_2, vertex_3}


def test_add_directed_edge() -> None:
    graph = LocalGraph()

    vertex_1 = Vertex("t", VertexType.TAG)
    vertex_2 = Vertex("u", VertexType.USER)
    vertex_3 = Vertex("i", VertexType.IMAGE)

    for vertex in [vertex_1, vertex_2, vertex_3]:
        with pytest.raises(KeyError):
            graph.out_neighbors(vertex)
        with pytest.raises(KeyError):
            graph.in_neighbors(vertex)

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
