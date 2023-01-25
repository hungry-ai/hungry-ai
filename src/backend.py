from users import UserService
from images import ImageService
from recommender import RecommenderService
from reviews import ReviewService
from graph import GraphService
from topics import TopicService
from db import UserDB, ImageDB, ReviewDB, GraphDB, TopicDB


class Backend:
    def __init__(self) -> None:
        graph_db = GraphDB("graph.csv")
        graph_service = GraphService(graph_db)

        topic_db = TopicDB("topics.csv")
        topic_service = TopicService(topic_db)

        user_db = UserDB("users.csv")
        self.user_service = UserService(user_db)

        image_db = ImageDB("images.csv")
        self.image_service = ImageService(image_db, topic_service, graph_service)

        review_db = ReviewDB("reviews.csv")
        self.review_service = ReviewService(review_db, graph_service)

        self.recommender_service = RecommenderService(graph_service, self.image_service)
