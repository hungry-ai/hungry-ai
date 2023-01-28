import datetime
from pathlib import Path

import pytest
from src.db import (
    GraphDB,
    Image,
    ImageDB,
    Recommendation,
    RecommendationDB,
    Review,
    ReviewDB,
    Topic,
    TopicDB,
    User,
    UserDB,
)
from src.graph import GraphService
from src.images import ImageService
from src.recommender import RecommenderService
from src.reviews import ReviewService
from src.topics import TopicService
from src.users import UserService, hash

# DBs


@pytest.fixture(scope="module")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def image_db(root: Path) -> ImageDB:
    return ImageDB(root / "images.csv")


@pytest.fixture(scope="module")
def topic_db(root: Path) -> TopicDB:
    return TopicDB(root / "topics.csv")


@pytest.fixture(scope="module")
def user_db(root: Path) -> UserDB:
    return UserDB(root / "users.csv")


@pytest.fixture(scope="module")
def review_db(root: Path) -> ReviewDB:
    return ReviewDB(root / "reviews.csv")


@pytest.fixture(scope="module")
def graph_db(root: Path) -> GraphDB:
    return GraphDB(root / "graph.csv")


@pytest.fixture(scope="module")
def recommendation_db(root: Path) -> RecommendationDB:
    return RecommendationDB(root / "recommendations.csv")


# services


@pytest.fixture(scope="module")
def graph_service(graph_db: GraphDB) -> GraphService:
    return GraphService(graph_db)


@pytest.fixture(scope="module")
def topic_service(topic_db: TopicDB) -> TopicService:
    topic_db.insert(Topic("t1", "hello world"))
    topic_db.insert(Topic("t2", "hi"))

    return TopicService(topic_db)


@pytest.fixture(scope="module")
def image_service(
    image_db: ImageDB, topic_service: TopicService, graph_service: GraphService
) -> ImageService:
    return ImageService(image_db, topic_service, graph_service)


@pytest.fixture(scope="module")
def user_service(user_db: UserDB) -> UserService:
    user_db.insert(User("u1", "a@gmail.com", hash("a")))

    return UserService(user_db)


@pytest.fixture(scope="module")
def review_service(review_db: ReviewDB, graph_db: GraphDB) -> ReviewService:
    review_db.insert(Review("r1", "u1", "i1", 5, datetime.datetime.now()))
    review_db.insert(Review("r2", "u2", "i2", 5, datetime.datetime.now()))

    graph_service = GraphService(graph_db)

    return ReviewService(review_db, graph_service)


@pytest.fixture(scope="module")
def recommender_service(
    recommendation_db: RecommendationDB,
    graph_service: GraphService,
    review_service: ReviewService,
) -> RecommenderService:
    graph_service.add_image_edge("i1", "t1", 0.5)
    graph_service.add_image_edge("i1", "t2", 0.5)
    graph_service.add_image_edge("i2", "t1", 0.5)

    graph_service.add_user_edge("u1", "t1", 5.0)
    graph_service.add_user_edge("u2", "t1", 5.0)
    graph_service.add_user_edge("u2", "t2", 1.0)

    return RecommenderService(recommendation_db, graph_service, review_service)


# random


@pytest.fixture(scope="module")
def url() -> str:
    return "https://i5.walmartimages.com/asr/5939a6fa-a0d6-431c-88c6-b4f21608e4be.f7cd0cc487761d74c69b7731493c1581.jpeg"
