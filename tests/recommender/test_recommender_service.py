from src.recommender import RecommenderService


def test_predict_ratings(recommender_service: RecommenderService) -> None:
    for user_id in ["cody", "alex", "younes"]:
        for image_id in ["tonkotsu", "chicken_noodle", "sushi"]:
            rating = recommender_service.predict_rating(user_id, image_id)
            assert 1.0 <= rating <= 5.0


def test_get_recommendations(recommender_service: RecommenderService) -> None:
    for user_id in ["cody", "alex", "younes"]:
        for num_recs in [0, 1, 2, 20]:
            recommendations = recommender_service.get_recommendations(user_id, num_recs)
            assert len(recommendations) <= num_recs
            assert all(
                isinstance(recommendation, str) for recommendation in recommendations
            )
            assert all(
                recommendation in ("tonkotsu", "chicken_noodle")
                for recommendation in recommendations
            )
