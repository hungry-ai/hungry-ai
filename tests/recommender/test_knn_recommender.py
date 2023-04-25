import pytest

from src.graph import LocalGraph
from src.images import Image
from src.recommender import KNNRecommender
from src.tags import Tag


@pytest.fixture(scope="function")
def recommender(japanese: Tag, soup: Tag, ramen: Tag) -> KNNRecommender:
    graph = LocalGraph()

    soup_vertex = graph.add_tag(soup.id)
    ramen_vertex = graph.add_tag(ramen.id)
    japanese_vertex = graph.add_tag(japanese.id)

    graph.add_edge(ramen_vertex, soup_vertex)
    graph.add_edge(ramen_vertex, japanese_vertex)

    tags = [japanese, soup, ramen]

    return KNNRecommender(graph, tags)


def test_get_image_tag_weights(
    recommender: KNNRecommender, tonkotsu: Image, japanese: Tag, soup: Tag, ramen: Tag
) -> None:
    image_tag_weights = recommender.get_image_tag_weights(tonkotsu)

    assert len(image_tag_weights) != 0
    assert set(image_tag_weights.keys()).issubset({japanese, soup, ramen})
    assert all(0 <= p <= 1 for p in image_tag_weights.values())


def test_add_image_edge(recommender: KNNRecommender, tonkotsu: Image) -> None:
    assert len(recommender.graph.images) == 0
    recommender.add_image(tonkotsu)
    assert len(recommender.graph.images) == 1


def test_get_closest_images(recommender: KNNRecommender) -> None:
    assert False
