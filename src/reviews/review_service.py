import datetime
from collections import defaultdict
from uuid import uuid4

from ..images import Image
from ..users import User
from .reviews import Review


class ReviewService:
    def __init__(self) -> None:
        self.reviews: dict[User, list[Review]] = defaultdict(list)

    def add_review(self, user: User, image: Image, rating: int) -> Review:
        review_id = str(uuid4())
        timestamp = datetime.datetime.now()
        review = Review(review_id, user, image, rating, timestamp)

        self.reviews[user].append(review)

        return review

    def get_reviews(self, user: User) -> list[Review]:
        return self.reviews[user]
