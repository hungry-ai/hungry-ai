import pytest
from src.db import Review
from src.reviews import ReviewService


def test_add_review(review_service: ReviewService) -> None:
    review_service.add_review("u1", "i1", 5)
    review_service.add_review("u1", "i2", 5)

    with pytest.raises(KeyError):
        review_service.add_review("u2", "i3", 5)


def test_get_reviews(review_service: ReviewService) -> None:
    review_service.add_review("u1", "i1", 5)
    review_service.add_review("u1", "i2", 5)

    reviews = review_service.get_reviews("u1")

    assert isinstance(reviews, list)
    assert len(reviews) > 0
    assert all(isinstance(review, Review) for review in reviews)
    assert all(review.user_id == "u1" for review in reviews)

    reviews = review_service.get_reviews("u2")

    assert isinstance(reviews, list)
    assert len(reviews) == 0
