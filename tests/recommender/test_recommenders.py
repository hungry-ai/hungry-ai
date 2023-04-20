from collections import defaultdict

import pytest

from src.images import Image
from src.recommender import Recommender
from src.reviews import Review
from src.users import User


class TestRecommender(Recommender):
    def __init__(self) -> None:
        self.user_ids: list[str] = []
        self.image_ids: list[str] = []
        self.reviews: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add_user(self, user: User) -> None:
        self.user_ids.append(user.id)

    def add_image(self, image: Image) -> None:
        self.image_ids.append(image.id)

    def add_review(self, review: Review) -> None:
        user_id = review.user.id
        image_id = review.image.id
        rating = review.rating
        self.reviews[user_id][image_id].append(rating)

    def get_recommendations(self, user: User, num_recs: int) -> list[str]:
        return self.image_ids[:num_recs]


@pytest.fixture(scope="function")
def test_recommender() -> TestRecommender:
    return TestRecommender()


ALL_RECOMMENDER_FIXTURES = [
    "test_recommender",
    "knn_recommender",
]


@pytest.mark.parametrize("recommender_fixture", ALL_RECOMMENDER_FIXTURES)
def test_adds(
    request: pytest.FixtureRequest,
    recommender_fixture: str,
    cody: User,
    tonkotsu: Image,
    cody_tonkotsu: Review,
) -> None:
    recommender = request.getfixturevalue(recommender_fixture)

    recommender.add_user(cody)
    recommender.add_image(tonkotsu)
    recommender.add_review(cody_tonkotsu)


@pytest.mark.parametrize("recommender_fixture", ALL_RECOMMENDER_FIXTURES)
def test_get_recommendations(
    request: pytest.FixtureRequest,
    recommender_fixture: str,
    cody: User,
    alex: User,
    younes: User,
) -> None:
    recommender = request.getfixturevalue(recommender_fixture)

    for user in [cody, alex, younes]:
        for num_recs in [0, 1, 2, 20]:
            recommendations = recommender.get_recommendations(user, num_recs)
            assert len(recommendations) <= num_recs
            assert all(
                isinstance(recommendation, str) for recommendation in recommendations
            )
            assert all(
                recommendation in ("_tonkotsu", "_chicken_noodle")
                for recommendation in recommendations
            )
