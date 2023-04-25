from abc import ABCMeta, abstractmethod

from ..images import Image
from ..reviews import Review
from ..users import User


class Recommender(metaclass=ABCMeta):
    @abstractmethod
    def add_user(self, user: User) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_image(self, image: Image) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_review(self, review: Review) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_recommendations(self, user: User, num_recs: int) -> list[str]:
        raise NotImplementedError
