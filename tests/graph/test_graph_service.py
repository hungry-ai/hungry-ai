import pytest

from src.graph import GraphService


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

    graph_service.add_image_edge("i1", "t1", 0.0)
    graph_service.add_image_edge("i2", "t2", 0.5)
    graph_service.add_image_edge("i3", "t3", 1.0)
    graph_service.add_image_edge("i1", "t1", 0.5)

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

    graph_service.add_user_edge("u1", "i1", 5.0)
    graph_service.add_user_edge("u1", "i2", 5.0)
    graph_service.add_user_edge("u2", "i1", 5.0)

    with pytest.raises(ValueError):
        graph_service.add_user_edge("u1", "i1", 0.5)

    with pytest.raises(ValueError):
        graph_service.add_user_edge("u1", "i1", 6.0)

    with pytest.raises(KeyError):
        graph_service.add_user_edge("u1", "i3", 5.0)

    with pytest.raises(KeyError):
        graph_service.add_user_edge("u3", "i1", 1.0)
