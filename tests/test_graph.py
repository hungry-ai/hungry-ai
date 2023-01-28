import datetime

import pytest
from src.db import Graph, Review
from src.graph import GraphService


def test_add_image_edge(graph_service: GraphService) -> None:
    graph_service.add_image_edge("i1", "t1", 0.0)
    graph_service.add_image_edge("i2", "t2", 0.5)
    graph_service.add_image_edge("i3", "t3", 1.0)
    graph_service.add_image_edge("i1", "t1", 0.5)

    with pytest.raises(ValueError):
        graph_service.add_image_edge("i1", "t1", -1.0)

    with pytest.raises(ValueError):
        graph_service.add_image_edge("i1", "t1", 2.0)


def test_add_user_edge(graph_service: GraphService) -> None:
    graph_service.add_image_edge("i1", "t1", 0.5)
    graph_service.add_image_edge("i2", "t2", 0.5)

    graph_service.add_user_edge("u1", "i1", 5.0)
    graph_service.add_user_edge("u1", "i2", 5.0)
    graph_service.add_user_edge("u2", "i1", 5.0)

    with pytest.raises(ValueError):
        graph_service.add_user_edge("u1", "i1", 0.5)

    with pytest.raises(ValueError):
        graph_service.add_user_edge("u1", "i1", 6.0)

    with pytest.raises(ValueError):
        graph_service.add_user_edge("u1", "i3", 5.0)


def test_predict_ratings(graph_service: GraphService) -> None:
    graph_service.add_image_edge("i1", "t1", 0.5)
    graph_service.add_image_edge("i2", "t2", 0.5)
    graph_service.add_image_edge("i3", "t3", 0.5)

    graph_service.add_user_edge("u1", "i1", 5.0)
    graph_service.add_user_edge("u2", "i1", 5.0)
    graph_service.add_user_edge("u2", "i2", 1.0)

    ratings = graph_service.predict_ratings("u1")
    assert isinstance(ratings, dict)
    assert all(isinstance(image_id, str) for image_id in ratings)
    assert all(isinstance(rating, float) for rating in ratings.values())
    assert set(ratings.keys()) == {"i1", "i2", "i3"}
    assert all(1.0 <= rating <= 5.0 for rating in ratings.values())
