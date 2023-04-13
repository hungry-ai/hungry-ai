import datetime
from pathlib import Path

import pytest

from src.backend import Backend
from src.db import (
    EdgeDB,
    Image,
    ImageDB,
    Recommendation,
    RecommendationDB,
    Review,
    ReviewDB,
    Tag,
    TagDB,
    User,
    UserDB,
)
from src.frontend import Frontend
from src.graph import GraphService
from src.images import ImageService
from src.recommender import RecommenderService
from src.reviews import ReviewService
from src.tags import TagService
from src.users import UserService, hash


# DBs


@pytest.fixture(scope="function")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="function")
def frontend(root: Path) -> Frontend:
    backend = Backend(root)
    return Frontend(backend)


@pytest.fixture(scope="function")
def image_db(root: Path) -> ImageDB:
    return ImageDB(root / "images.csv")


@pytest.fixture(scope="function")
def tag_db(root: Path) -> TagDB:
    return TagDB(root / "tags.csv")


@pytest.fixture(scope="function")
def user_db(root: Path) -> UserDB:
    return UserDB(root / "users.csv")


@pytest.fixture(scope="function")
def review_db(root: Path) -> ReviewDB:
    return ReviewDB(root / "reviews.csv")


@pytest.fixture(scope="function")
def edge_db(root: Path) -> EdgeDB:
    return EdgeDB(root / "edges.csv")


@pytest.fixture(scope="function")
def recommendation_db(root: Path) -> RecommendationDB:
    return RecommendationDB(root / "recommendations.csv")


# services


@pytest.fixture(scope="function")
def graph_service(edge_db: EdgeDB) -> GraphService:
    return GraphService(edge_db)


@pytest.fixture(scope="function")
def tag_service(tag_db: TagDB) -> TagService:
    tag_db.insert(Tag("t1", "hello world"))
    tag_db.insert(Tag("t2", "hi"))

    return TagService(tag_db)


@pytest.fixture(scope="function")
def image_service(
    image_db: ImageDB, tag_service: TagService, graph_service: GraphService
) -> ImageService:
    return ImageService(image_db, tag_service, graph_service)


@pytest.fixture(scope="function")
def user_service(user_db: UserDB) -> UserService:
    user_db.insert(User("u1", "a@gmail.com", hash("a")))

    return UserService(user_db)


@pytest.fixture(scope="function")
def review_service(review_db: ReviewDB, graph_service: GraphService) -> ReviewService:
    graph_service.add_image_edge("i1", "t1", 0.8)
    graph_service.add_image_edge("i2", "t1", 0.8)
    graph_service.add_image_edge("i2", "t2", 0.5)

    return ReviewService(review_db, graph_service)


@pytest.fixture(scope="function")
def recommender_service(
    recommendation_db: RecommendationDB,
    graph_service: GraphService,
) -> RecommenderService:
    graph_service.add_image_edge("i1", "t1", 0.5)
    graph_service.add_image_edge("i1", "t2", 0.5)
    graph_service.add_image_edge("i2", "t1", 0.5)

    graph_service.add_user_edge("u1", "i1", 5.0)
    graph_service.add_user_edge("u2", "i1", 5.0)
    graph_service.add_user_edge("u2", "i2", 1.0)

    return RecommenderService(recommendation_db, graph_service)


# random


@pytest.fixture(scope="module")
def url() -> str:
    return "https://i5.walmartimages.com/asr/5939a6fa-a0d6-431c-88c6-b4f21608e4be.f7cd0cc487761d74c69b7731493c1581.jpeg"
