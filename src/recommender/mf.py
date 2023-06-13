import json
import numpy as np
from pathlib import Path
from typing import Self

from ..tags import Tag
from ..images import Image, TaggedImage
from ..reviews import Review
from ..users import User
from .recommender import Recommender


class MFRecommender(Recommender):
    def __init__(
        self,
        tag_weights: dict[Tag, np.ndarray],
        x_default: np.ndarray,
        d: int,
        alpha: float,
    ) -> None:
        pass

    def add_user(self, user: User) -> None:
        raise NotImplementedError

    def add_image(self, image: Image) -> None:
        if not isinstance(image, TaggedImage):
            raise ValueError("MFRecommender only supports adding TaggedImages")

        raise NotImplementedError

    def add_review(self, review: Review) -> None:
        raise NotImplementedError

    def predict_rating(self, user: User, image: Image) -> float:
        raise NotImplementedError

    def get_recommendations(self, user: User, num_recs: int) -> list[Image]:
        raise NotImplementedError

    def save(self, json_path: Path) -> None:
        # mf_{id}.json
        # tags_weights_{id}.csv
        pass

    @staticmethod
    def load(self, json_path: Path) -> Self:
        pass
