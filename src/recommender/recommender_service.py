from ..images import Image
from ..reviews import Review
from ..users import User
from .recommender import Recommender


class RecommenderService:
    def __init__(self, recommender: Recommender) -> None:
        self.recommender = recommender

    def add_user(self, user: User) -> None:
        self.recommender.add_user(user)

    def add_image(self, image: Image) -> None:
        self.recommender.add_image(image)

    def add_review(self, review: Review) -> None:
        self.recommender.add_review(review)

    def get_recommendations(self, user: User, num_recs: int) -> list[str]:
        return self.recommender.get_recommendations(user, num_recs=num_recs)
