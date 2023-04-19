from ..db import RecommendationDB
from .recommender import Recommender


class RecommenderService:
    def __init__(
        self,
        recommendation_db: RecommendationDB,
        recommender: Recommender,
    ) -> None:
        self.recommendation_db = recommendation_db
        self.recommender = recommender

    def predict_rating(self, user_id: str, image_id: str) -> float:
        return self.recommender.predict_rating(user_id, image_id)

    def get_recommendations(self, user_id: str, num_recs: int) -> list[str]:
        return self.recommender.get_recommendations(user_id, num_recs)
