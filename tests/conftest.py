from pathlib import Path

import pytest

from src.backend import Backend
from src.db import EdgeDB, ImageDB, RecommendationDB, ReviewDB, Tag, TagDB, User, UserDB
from src.frontend import Frontend
from src.graph import GraphService, LocalGraph
from src.images import ImageService
from src.recommender import Recommender, KNNRecommender, RecommenderService
from src.reviews import ReviewService
from src.tags import TagService
from src.users import UserService

# DBs


@pytest.fixture(scope="function")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="function")
def backend(root: Path) -> Backend:
    return Backend(root)


@pytest.fixture(scope="function")
def frontend(backend: Backend) -> Frontend:
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
def graph_service() -> GraphService:
    graph = LocalGraph()

    graph.add_tag("t1")
    graph.add_tag("t2")
    graph.add_tag("t3")

    return GraphService(graph)


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
def user_service(user_db: UserDB, graph_service: GraphService) -> UserService:
    user_db.insert(User("u1", "u1_inst"))
    graph_service.add_user("u1")

    return UserService(user_db, graph_service)


@pytest.fixture(scope="function")
def review_service(review_db: ReviewDB, graph_service: GraphService) -> ReviewService:
    graph_service.add_user("u1")

    graph_service.add_image("i1")
    graph_service.add_image("i2")

    graph_service.add_image_edge("i1", "t1", 0.8)
    graph_service.add_image_edge("i2", "t1", 0.8)
    graph_service.add_image_edge("i2", "t2", 0.5)

    return ReviewService(review_db, graph_service)


@pytest.fixture(scope="function")
def recommender_service(recommendation_db: RecommendationDB) -> RecommenderService:
    graph = LocalGraph()

    soup = graph.add_tag("soup")
    ramen = graph.add_tag("ramen")
    japanese = graph.add_tag("japanese")

    graph.add_edge(ramen, soup)
    graph.add_edge(ramen, japanese)

    tonkotsu = graph.add_image("tonkotsu")
    chicken_noodle = graph.add_image("chicken_noodle")

    graph.add_edge(tonkotsu, soup, 1.0, directed=False)
    graph.add_edge(tonkotsu, ramen, 1.0, directed=False)
    graph.add_edge(tonkotsu, japanese, 1.0, directed=False)
    graph.add_edge(chicken_noodle, soup, 1.0, directed=False)

    cody = graph.add_user("cody")
    alex = graph.add_user("alex")

    graph.add_edge(cody, tonkotsu, 5.0, directed=False)
    graph.add_edge(alex, tonkotsu, 5.0, directed=False)
    graph.add_edge(alex, chicken_noodle, 1.0, directed=False)

    recommender = KNNRecommender(graph)

    return RecommenderService(recommendation_db, recommender)


# random


@pytest.fixture(scope="module")
def url() -> str:
    return "https://i5.walmartimages.com/asr/5939a6fa-a0d6-431c-88c6-b4f21608e4be.f7cd0cc487761d74c69b7731493c1581.jpeg"
