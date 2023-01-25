from __future__ import annotations

from db import Image
from graph import GraphService
from images import ImageService


class RecommenderService:
    def __init__(
        self, graph_service: GraphService, image_service: ImageService
    ) -> None:
        self.graph_service = graph_service
        self.image_service = image_service
        self.rated = set()

    def recommend(self, user_id: str) -> None | Image:
        ratings = self.graph_service.predict_ratings(user_id)
        if len(self.rated) == len(ratings):
            self.rated.clear()

        for rating, image_id in sorted(
            [(v, k) for k, v in ratings.items()], reverse=True
        ):
            if image_id in self.rated:
                continue
            return self.image_service.get_image(image_id)
