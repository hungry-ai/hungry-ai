from .graph import LocalGraph
from .images import Image, ImageService
from .recommender import KNNRecommender, RecommenderService
from .reviews import Review, ReviewService
from .tags import Tag
from .users import User, UserService


class Backend:
    def __init__(self) -> None:
        self.user_service = UserService()

        self.image_service = ImageService()

        self.review_service = ReviewService()

        graph = LocalGraph()
        tags: list[Tag] = []
        recommender = KNNRecommender(graph, tags)
        self.recommender_service = RecommenderService(recommender)

    def get_user(self, instagram_username: str) -> User:
        return self.user_service.get_user(instagram_username)

    def add_user(self, instagram_username: str) -> User:
        user = self.user_service.add_user(instagram_username)
        self.recommender_service.add_user(user)
        return user

    def add_image(self, url: str) -> Image:
        image = self.image_service.add_image(url)
        self.recommender_service.add_image(image)
        return image

    def add_review(self, user: User, image: Image, rating: int) -> Review:
        review = self.review_service.add_review(user, image, rating)
        self.recommender_service.add_review(review)
        return review

    def get_reviews(self, user: User) -> list[Review]:
        reviews = self.review_service.get_reviews(user)
        return reviews

    def get_recommendations(self, user: User, num_recs: int) -> list[Image]:
        image_ids = self.recommender_service.get_recommendations(user, num_recs)
        recommendations = [
            self.image_service.get_image(image_id) for image_id in image_ids
        ]
        return recommendations
