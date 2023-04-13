import pytest

from src.graph import LocalGraph, Vertex


def test_graph():
    graph_1 = LocalGraph()
    assert 0 == len(graph_1.vertices)
    assert 0 == len(graph_1.edges)

    with pytest.raises(KeyError):
        result = graph_1.add_edge(Vertex(1, 0), Vertex(2, 0))

    vertex_1 = Vertex(1, 1)
    graph_1.add_vertex(vertex_1)
    vertex_2 = Vertex(10392, 0)
    graph_1.add_vertex(vertex_2)
    vertex_3 = Vertex(30000, 2)
    graph_1.add_vertex(vertex_3)
    # assert 0 == vertex_1
    # assert 1 == vertex_2
    # assert 2 == vertex_3

    edge_1 = graph_1.add_edge(vertex_1, vertex_2, 1)
    egde_2 = graph_1.add_edge(vertex_1, vertex_3, 3)
    # assert edge_1[2] == 1
    # assert egde_2[2] == 3
