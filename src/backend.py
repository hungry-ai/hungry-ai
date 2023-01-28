from pathlib import Path

from .db import GraphDB, ImageDB, RecommendationDB, ReviewDB, TopicDB, UserDB
from .graph import GraphService
from .images import ImageService
from .recommender import RecommenderService
from .reviews import ReviewService
from .topics import TopicService
from .users import UserService


class Backend:
    def __init__(self, root: Path = Path(__file__).parent.parent / "data") -> None:
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

        recommendation_db = RecommendationDB(root / "recommendations.csv")
        self.recommender_service = RecommenderService(
            recommendation_db, graph_service, self.review_service
        )
