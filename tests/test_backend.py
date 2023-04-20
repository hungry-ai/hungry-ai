import pytest

from src.backend import Backend
from src.images import Image
from src.reviews import Review
from src.users import User


def test_user(backend: Backend) -> None:
    username = "cody"

    with pytest.raises(KeyError):
        backend.get_user(username)

    cody = backend.add_user(username)
    assert isinstance(cody, User)
    assert cody.instagram_username == username

    assert backend.get_user(username) == cody


def test_add_image(backend: Backend, url: str) -> None:
    banana = backend.add_image(url)
    assert isinstance(banana, Image)
    assert banana.url == url


def test_add_review(backend: Backend, url: str) -> None:
    cody = backend.add_user("cody")
    banana = backend.add_image(url)

    cody_banana = backend.add_review(cody, banana, 3)
    assert isinstance(cody_banana, Review)
    assert cody_banana.user == cody
    assert cody_banana.user.instagram_username == "cody"
    assert cody_banana.image == banana
    assert cody_banana.image.url == url
    assert cody_banana.rating == 3.0


def test_get_recommendations(backend: Backend, url: str) -> None:
    cody = backend.add_user("cody")
    backend.add_image(url)

    recommendations = backend.get_recommendations(cody, 20)

    assert len(recommendations) <= 20
    assert all(isinstance(recommendation, Image) for recommendation in recommendations)
