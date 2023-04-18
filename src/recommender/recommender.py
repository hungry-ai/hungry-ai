from abc import ABCMeta, abstractmethod
from ..graph import Graph


class Recommender(metaclass=ABCMeta):
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    @abstractmethod
    def predict_rating(self, user_id: str, image_id: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_recommendations(self, user_id: int, num_recs: int) -> list[str]:
        raise NotImplementedError
