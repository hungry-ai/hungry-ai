from users import UserService
from images import ImageService
from recommender import RecommenderService
from reviews import ReviewService
from graph import GraphService
from topics import TopicService
from db import UserDB, ImageDB, ReviewDB, GraphDB, TopicDB

from pathlib import Path


class Backend:
    def __init__(self) -> None:
        root = Path(__file__).parent.parent / "data"
        root.mkdir(exist_ok=True)

        graph_db = GraphDB(root / "graph.csv")
        graph_service = GraphService(graph_db)

        topic_db = TopicDB(root / "topics.csv")
        topic_service = TopicService(topic_db)

        user_db = UserDB(root / "users.csv")
        self.user_service = UserService(user_db)

        image_db = ImageDB(root / "images.csv")
        self.image_service = ImageService(image_db, topic_service, graph_service)

        review_db = ReviewDB(root / "reviews.csv")
        self.review_service = ReviewService(review_db, graph_service)

        self.recommender_service = RecommenderService(graph_service, self.image_service)
