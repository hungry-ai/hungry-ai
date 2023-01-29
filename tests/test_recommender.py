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
    review_db: ReviewDB,
    graph_service: GraphService,
) -> None:
    review_service = ReviewService(review_db, graph_service)
    recommender_service = RecommenderService(recommendation_db, review_service)

    image_id = recommender_service.recommend("u1")

    assert image_id is None
