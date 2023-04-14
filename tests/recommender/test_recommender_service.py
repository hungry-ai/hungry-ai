from src.db import Recommendation, RecommendationDB, ReviewDB
from src.graph import GraphService
from src.recommender import RecommenderService
from src.reviews import ReviewService


def test_recommend(recommender_service: RecommenderService) -> None:
    image_id = recommender_service.recommend("u1")

    assert image_id is not None
    assert isinstance(image_id, str)

    image_id = recommender_service.recommend("u1")

    assert image_id is not None
    assert isinstance(image_id, str)

    image_id = recommender_service.recommend("u1")

    assert image_id is not None
    assert isinstance(image_id, str)


def test_recommend_empty(
    recommendation_db: RecommendationDB,
    graph_service: GraphService,
) -> None:
    recommender_service = RecommenderService(recommendation_db, graph_service)

    image_id = recommender_service.recommend("u1")

    assert image_id is None


def test_predict_ratings(graph_service: GraphService) -> None:
    graph_service.add_image("i1")
    graph_service.add_image("i2")
    graph_service.add_image("i3")

    graph_service.add_image_edge("i1", "t1", 0.5)
    graph_service.add_image_edge("i2", "t2", 0.5)
    graph_service.add_image_edge("i3", "t3", 0.5)

    graph_service.add_user("u1")
    graph_service.add_user("u2")

    graph_service.add_user_edge("u1", "i1", 5.0)
    graph_service.add_user_edge("u2", "i1", 5.0)
    graph_service.add_user_edge("u2", "i2", 1.0)

    ratings = graph_service.predict_ratings("u1")
    assert isinstance(ratings, dict)
    assert all(isinstance(image_id, str) for image_id in ratings)
    assert all(isinstance(rating, float) for rating in ratings.values())
    assert set(ratings.keys()) == {"i1", "i2", "i3"}
    assert all(1.0 <= rating <= 5.0 for rating in ratings.values())

    graph_service.predict_ratings("u3")
