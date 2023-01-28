from src.db import Review
from src.reviews import ReviewService


def test_add_review(review_service: ReviewService) -> None:
    review_service.add_review("u3", "i3", 5)


def test_get_reviews(review_service: ReviewService) -> None:
    reviews = review_service.get_reviews("u1")

    assert isinstance(reviews, list)
    assert len(reviews) > 0
    assert all(isinstance(review, Review) for review in reviews)
    assert all(review.user_id == "u1" for review in reviews)
