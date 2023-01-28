from __future__ import annotations

from .db import Image, Recommendation, RecommendationDB
from .graph import GraphService
from .reviews import ReviewService


class RecommenderService:
    def __init__(
        self,
        recommendation_db: RecommendationDB,
        graph_service: GraphService,
        review_service: ReviewService,
    ) -> None:
        self.recommendation_db = recommendation_db
        self.graph_service = graph_service
        self.review_servies = review_service

        self.rated: set[str] = set()

    def recommend(self, user_id: str) -> None | str:
        ratings = self.graph_service.predict_ratings(user_id)
        if len(self.rated) == len(ratings):
            self.rated.clear()

        for rating, image_id in sorted(
            [(v, k) for k, v in ratings.items()], reverse=True
        ):
            if image_id in self.rated:
                continue
            return image_id

        return None
