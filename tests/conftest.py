import datetime
from pathlib import Path

import pytest

from src.backend import Backend
from src.frontend import Frontend
from src.graph import LocalGraph
from src.images import Image, ImageService
from src.recommender import KNNRecommender, RecommenderService
from src.reviews import Review, ReviewService
from src.tags import Tag
from src.users import User, UserService

# users


@pytest.fixture(scope="function")
def cody() -> User:
    return User("_cody", "cody")


@pytest.fixture(scope="function")
def alex() -> User:
    return User("_alex", "alex")


@pytest.fixture(scope="function")
def younes() -> User:
    return User("_younes", "younes")


# images


@pytest.fixture(scope="function")
def tonkotsu() -> Image:
    return Image(
        "_tonkotsu",
        "https://a0.cdn.japantravel.com/photo/31400-157114/1440x960!/tokyo-tonchin-ramen-157114.jpg",
    )


@pytest.fixture(scope="function")
def chicken_noodle() -> Image:
    return Image(
        "_chicken_noodle",
        "https://www.averiecooks.com/wp-content/uploads/2015/09/chickennoodlesoup-14.jpg",
    )


@pytest.fixture(scope="function")
def sushi() -> Image:
    return Image(
        "_sushi",
        "https://pyxis.nymag.com/v1/imgs/9e3/a79/80a32b4009084c8c3cdb7fff4f1bc7d407-sushi-nakazawa-03.rsocial.w1200.jpg",
    )


# tags


@pytest.fixture(scope="function")
def japanese() -> Tag:
    return Tag("_japanese", "japanese")


@pytest.fixture(scope="function")
def soup() -> Tag:
    return Tag("_soup", "soup")


@pytest.fixture(scope="function")
def ramen() -> Tag:
    return Tag("_ramen", "ramen")


# reviews


@pytest.fixture(scope="function")
def cody_tonkotsu(cody: User, tonkotsu: Image) -> Review:
    return Review("_cody_tonkotsu", cody, tonkotsu, 5, datetime.datetime(2023, 1, 1))


@pytest.fixture(scope="function")
def alex_tonkotsu(alex: User, tonkotsu: Image) -> Review:
    return Review("_alex_tonkotsu", alex, tonkotsu, 5, datetime.datetime(2023, 1, 1))


@pytest.fixture(scope="function")
def alex_chicken_noodle(alex: User, chicken_noodle: Image) -> Review:
    return Review(
        "_alex_chicken_noodle", alex, chicken_noodle, 1, datetime.datetime(2023, 1, 2)
    )


@pytest.fixture(scope="function")
def younes_sushi_1(younes: User, sushi: Image) -> Review:
    return Review("_younes_sushi_1", younes, sushi, 5, datetime.datetime(2023, 1, 1))


@pytest.fixture(scope="function")
def younes_sushi_2(younes: User, sushi: Image) -> Review:
    return Review("_younes_sushi_2", younes, sushi, 4, datetime.datetime(2023, 1, 2))


# recommenders


@pytest.fixture(scope="function")
def local_graph() -> LocalGraph:
    graph = LocalGraph()

    soup = graph.add_tag("_soup")
    ramen = graph.add_tag("_ramen")
    japanese = graph.add_tag("_japanese")

    graph.add_edge(ramen, soup)
    graph.add_edge(ramen, japanese)

    tonkotsu = graph.add_image("_tonkotsu")
    chicken_noodle = graph.add_image("_chicken_noodle")

    graph.add_edge(tonkotsu, soup, 1.0, directed=False)
    graph.add_edge(tonkotsu, ramen, 1.0, directed=False)
    graph.add_edge(tonkotsu, japanese, 1.0, directed=False)
    graph.add_edge(chicken_noodle, soup, 1.0, directed=False)

    cody = graph.add_user("_cody")
    alex = graph.add_user("_alex")

    graph.add_edge(cody, tonkotsu, 5.0, directed=False)
    graph.add_edge(alex, tonkotsu, 5.0, directed=False)
    graph.add_edge(alex, chicken_noodle, 1.0, directed=False)

    return graph


@pytest.fixture(scope="function")
def knn_recommender(
    local_graph: LocalGraph, japanese: Tag, soup: Tag, ramen: Tag
) -> KNNRecommender:
    return KNNRecommender(local_graph, [japanese, soup, ramen])


# services


@pytest.fixture(scope="function")
def image_service() -> ImageService:
    return ImageService()


@pytest.fixture(scope="function")
def user_service() -> UserService:
    return UserService()


@pytest.fixture(scope="function")
def review_service() -> ReviewService:
    return ReviewService()


@pytest.fixture(scope="function")
def recommender_service(knn_recommender: KNNRecommender) -> RecommenderService:
    return RecommenderService(knn_recommender)


# overall


@pytest.fixture(scope="function")
def root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="function")
def backend(root: Path) -> Backend:
    return Backend()


@pytest.fixture(scope="function")
def frontend(backend: Backend) -> Frontend:
    return Frontend(backend)


# random


@pytest.fixture(scope="function")
def url() -> str:
    return "https://images.heb.com/is/image/HEBGrocery/000377497"
