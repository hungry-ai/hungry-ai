from src.db import Recommendation, RecommendationDB
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
    review_service: ReviewService,
) -> None:
    recommender_service = RecommenderService(
        recommendation_db, graph_service, review_service
    )

    image_id = recommender_service.recommend("u1")
    assert image_id is None
