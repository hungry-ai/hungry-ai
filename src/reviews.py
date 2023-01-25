from db import Review, ReviewDB
from graph import GraphService

import datetime
from uuid import uuid4


class ReviewService:
    def __init__(self, review_db: ReviewDB, graph_service: GraphService) -> None:
        self.review_db = review_db
        self.graph_service = graph_service

    def review(self, user_id: str, image_id: str, rating: int) -> None:
        review_id = uuid4()
        timestamp = datetime.datetime.now()
        review = Review(review_id, user_id, image_id, rating, timestamp)
        self.review_db.insert(review)

        self.graph_service.add_user_edge(user_id, image_id, rating)
