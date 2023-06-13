from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Self

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
    def get_recommendations(self, user: User, num_recs: int) -> list[Image]:
        raise NotImplementedError

    @property
    def json_path(self) -> Path:
        name = self.__class__.__name__.lower().removesuffix("recommender")
        json_root = Path(__file__).parents[2] / "data" / "models"
        i = 0
        while (json_root / f"{name}_{i}.json").exists():
            i += 1
        return json_root / f"{name}_{i}.json"

    @abstractmethod
    def save(self, json_path: Path = Path(__file__)) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(self, json_path: Path = Path(__file__)) -> Self:
        raise NotImplementedError
