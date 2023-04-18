import pytest

from src.recommender import Recommender, KNNRecommender
from src.graph import LocalGraph, Graph, VertexType


class TestRecommender(Recommender):
    def __init__(self, graph: Graph) -> None:
        super().__init__(graph)

    def predict_rating(self, user_id: str, image_id: str) -> float:
        return 2.5

    def get_recommendations(self, user_id: int, num_recs: int) -> list[str]:
        images = [
            vertex.id
            for vertex in self.graph.vertices
            if vertex.type == VertexType.IMAGE
        ]
        return images[:num_recs]


@pytest.fixture(scope="function")
def graph() -> Graph:
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

    return graph


@pytest.fixture(scope="function")
def test_recommender(graph: Graph) -> TestRecommender:
    return TestRecommender(graph)


@pytest.fixture(scope="function")
def knn_recommender(graph: Graph) -> KNNRecommender:
    return KNNRecommender(graph)


ALL_RECOMMENDER_FIXTURES = [
    "test_recommender",
    "knn_recommender",
]


@pytest.mark.parametrize("recommender_fixture", ALL_RECOMMENDER_FIXTURES)
def test_predict_ratings(
    request: pytest.FixtureRequest, recommender_fixture: str
) -> None:
    recommender = request.getfixturevalue(recommender_fixture)

    for user_id in ["cody", "alex", "younes"]:
        for image_id in ["tonkotsu", "chicken_noodle", "sushi"]:
            rating = recommender.predict_rating(user_id, image_id)
            assert 1.0 <= rating <= 5.0


@pytest.mark.parametrize("recommender_fixture", ALL_RECOMMENDER_FIXTURES)
def test_get_recommendations(
    request: pytest.FixtureRequest, recommender_fixture: str
) -> None:
    recommender = request.getfixturevalue(recommender_fixture)

    for user_id in ["cody", "alex", "younes"]:
        for num_recs in [0, 1, 2, 20]:
            recommendations = recommender.get_recommendations(user_id, num_recs)
            assert len(recommendations) <= num_recs
            assert all(
                isinstance(recommendation, str) for recommendation in recommendations
            )
            assert all(
                recommendation in ("tonkotsu", "chicken_noodle")
                for recommendation in recommendations
            )
