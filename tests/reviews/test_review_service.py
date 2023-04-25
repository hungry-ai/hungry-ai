import datetime

from src.images import Image
from src.reviews import Review, ReviewService
from src.users import User


def test_review_service(
    review_service: ReviewService, cody: User, tonkotsu: Image
) -> None:
    before = datetime.datetime.now()
    cody_tonkotsu = review_service.add_review(cody, tonkotsu, 5)

    assert isinstance(cody_tonkotsu, Review)
    assert cody_tonkotsu.user == cody
    assert cody_tonkotsu.image == tonkotsu
    assert cody_tonkotsu.rating == 5
    assert before <= cody_tonkotsu.timestamp <= datetime.datetime.now()
