from src.db import Recommendation, RecommendationDB, ReviewDB
from src.graph import GraphService
from src.recommender import RecommenderService
from src.reviews import ReviewService


def test_predict_ratings(recommender_service: RecommenderService) -> None:
    predicted_ratings = [
        recommender_service.predict_rating("u1", "i1"),
        recommender_service.predict_rating("u1", "i2"),
        recommender_service.predict_rating("u2", "i1"),
        recommender_service.predict_rating("u2", "i2"),
        recommender_service.predict_rating("u3", "i1"),
        recommender_service.predict_rating("u3", "i2"),
    ]

    for rating in predicted_ratings:
        assert 1.0 <= rating <= 5.0


def test_get_recommendations(recommender_service: RecommenderService) -> None:
    recommendations_1 = recommender_service.get_recommendations("u1", 20)
    recommendations_2 = recommender_service.get_recommendations("u2", 20)
    recommendations_3 = recommender_service.get_recommendations("u3", 20)

    assert len(recommendations_1) >= 2
    assert len(recommendations_2) >= 2
    assert len(recommendations_3) == 0
