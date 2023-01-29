from pathlib import Path

from .db import EdgeDB, ImageDB, RecommendationDB, ReviewDB, TopicDB, UserDB
from .graph import GraphService
from .images import ImageService
from .recommender import RecommenderService
from .reviews import ReviewService
from .topics import TopicService
from .users import UserService


class Backend:
    def __init__(self, root: Path = Path(__file__).parent.parent / "data") -> None:
        root.mkdir(exist_ok=True)

        edge_db = EdgeDB(root / "edges.csv")
        graph_service = GraphService(edge_db)

        topic_db = TopicDB(root / "topics.csv")
        self.topic_service = TopicService(topic_db)

        user_db = UserDB(root / "users.csv")
        self.user_service = UserService(user_db)

        image_db = ImageDB(root / "images.csv")
        self.image_service = ImageService(image_db, self.topic_service, graph_service)

        review_db = ReviewDB(root / "reviews.csv")
        self.review_service = ReviewService(review_db, graph_service)

        recommendation_db = RecommendationDB(root / "recommendations.csv")
        self.recommender_service = RecommenderService(
            recommendation_db, self.review_service
        )
