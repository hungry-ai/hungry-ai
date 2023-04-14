import datetime
from uuid import uuid4

from ..db import Review, ReviewDB
from ..graph import GraphService


class ReviewService:
    def __init__(self, review_db: ReviewDB, graph_service: GraphService) -> None:
        self.review_db = review_db
        self.graph_service = graph_service

    def add_review(self, user_id: str, image_id: str, rating: int) -> None:
        review_id = str(uuid4())
        timestamp = datetime.datetime.now()
        review = Review(review_id, user_id, image_id, rating, timestamp)
        self.review_db.insert(review)

        self.graph_service.add_user_edge(user_id, image_id, rating)

    def get_reviews(self, user_id: str) -> list[Review]:
        return self.review_db.select(user_id=user_id)
