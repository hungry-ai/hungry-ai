import pytest

from src.graph import GraphService, Vertex, VertexType


def test_add_image(graph_service: GraphService) -> None:
    graph_service.add_image("i1")

    with pytest.warns():
        graph_service.add_image("i1")


def test_add_user(graph_service: GraphService) -> None:
    graph_service.add_user("u1")

    with pytest.warns():
        graph_service.add_user("u1")


def test_add_image_edge(graph_service: GraphService) -> None:
    graph_service.add_image("i1")
    graph_service.add_image("i2")
    graph_service.add_image("i3")

    i1 = Vertex("i1", VertexType.IMAGE)
    i2 = Vertex("i2", VertexType.IMAGE)
    i3 = Vertex("i3", VertexType.IMAGE)
    t1 = Vertex("t1", VertexType.TAG)
    t2 = Vertex("t2", VertexType.TAG)
    t3 = Vertex("t3", VertexType.TAG)

    graph_service.add_image_edge("i1", "t1", 0.0)
    assert i1 in graph_service.graph.out_neighbors(t1)
    assert t1 in graph_service.graph.out_neighbors(i1)
    assert i1 in graph_service.graph.in_neighbors(t1)
    assert t1 in graph_service.graph.in_neighbors(i1)
    graph_service.add_image_edge("i2", "t2", 0.5)
    assert i2 in graph_service.graph.out_neighbors(t2)
    assert t2 in graph_service.graph.out_neighbors(i2)
    assert i2 in graph_service.graph.in_neighbors(t2)
    assert t2 in graph_service.graph.in_neighbors(i2)
    graph_service.add_image_edge("i3", "t3", 1.0)
    assert i3 in graph_service.graph.out_neighbors(t3)
    assert t3 in graph_service.graph.out_neighbors(i3)
    assert i3 in graph_service.graph.in_neighbors(t3)
    assert t3 in graph_service.graph.in_neighbors(i3)
    graph_service.add_image_edge("i1", "t1", 0.5)
    assert i1 in graph_service.graph.out_neighbors(t1)
    assert t1 in graph_service.graph.out_neighbors(i1)
    assert i1 in graph_service.graph.in_neighbors(t1)
    assert t1 in graph_service.graph.in_neighbors(i1)

    with pytest.raises(ValueError):
        graph_service.add_image_edge("i1", "t1", -1.0)

    with pytest.raises(ValueError):
        graph_service.add_image_edge("i1", "t1", 2.0)

    with pytest.raises(KeyError):
        graph_service.add_image_edge("i4", "t1", 1.0)

    with pytest.raises(KeyError):
        graph_service.add_image_edge("i1", "t4", 1.0)


def test_add_user_edge(graph_service: GraphService) -> None:
    graph_service.add_image("i1")
    graph_service.add_image("i2")

    graph_service.add_image_edge("i1", "t1", 0.5)
    graph_service.add_image_edge("i2", "t2", 0.5)

    graph_service.add_user("u1")
    graph_service.add_user("u2")

    i1 = Vertex("i1", VertexType.IMAGE)
    i2 = Vertex("i2", VertexType.IMAGE)
    u1 = Vertex("u1", VertexType.USER)
    u2 = Vertex("u2", VertexType.USER)

    graph_service.add_user_edge("u1", "i1", 5.0)
    assert i1 in graph_service.graph.out_neighbors(u1)
    assert u1 in graph_service.graph.out_neighbors(i1)
    assert i1 in graph_service.graph.in_neighbors(u1)
    assert u1 in graph_service.graph.in_neighbors(i1)
    graph_service.add_user_edge("u1", "i2", 5.0)
    assert i2 in graph_service.graph.out_neighbors(u1)
    assert u1 in graph_service.graph.out_neighbors(i2)
    assert i2 in graph_service.graph.in_neighbors(u1)
    assert u1 in graph_service.graph.in_neighbors(i2)
    graph_service.add_user_edge("u2", "i1", 5.0)
    assert i1 in graph_service.graph.out_neighbors(u2)
    assert u2 in graph_service.graph.out_neighbors(i1)
    assert i1 in graph_service.graph.in_neighbors(u2)
    assert u2 in graph_service.graph.in_neighbors(i1)

    with pytest.raises(ValueError):
        graph_service.add_user_edge("u1", "i1", 0.5)

    with pytest.raises(ValueError):
        graph_service.add_user_edge("u1", "i1", 6.0)

    with pytest.raises(KeyError):
        graph_service.add_user_edge("u1", "i3", 5.0)

    with pytest.raises(KeyError):
        graph_service.add_user_edge("u3", "i1", 1.0)
