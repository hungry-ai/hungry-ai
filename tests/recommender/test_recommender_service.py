from src.images import Image
from src.recommender import RecommenderService
from src.reviews import Review
from src.users import User


def test_recommender_service(
    recommender_service: RecommenderService,
    cody: User,
    tonkotsu: Image,
    cody_tonkotsu: Review,
) -> None:
    recommender_service.add_user(cody)
    recommender_service.add_image(tonkotsu)
    recommender_service.add_review(cody_tonkotsu)

    recommendations = recommender_service.get_recommendations(cody, 20)
    assert all(isinstance(recommendation, str) for recommendation in recommendations)
