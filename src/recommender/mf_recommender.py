import heapq
import pandas as pd  # type: ignore[import]
import numpy as np

from .recommender import Recommender
from ..graph import LocalGraph, Graph, Vertex, VertexType
from ..tags import WordEmbedding, generate_tags_graph


class MFRecommender(Recommender):
    def __init__(
        self,
        graph: Graph,
        X: np.ndarray,
        Y: np.ndarray,
        user_indices: dict[str, int],
        image_indices: dict[str, int],
    ) -> None:
        super().__init__(graph)
        self.X = X
        self.Y = Y
        self.user_indices = user_indices
        self.image_indices = image_indices
        self.avgX = np.mean(X, axis=0)
        self.avgY = np.mean(Y, axis=0)
        self.N = X.shape[0]
        self.M = Y.shape[0]

    def predict_rating(self, user_id: str, image_id: str) -> float:
        user_index = self.user_indices.get(user_id, -1)
        image_index = self.image_indices.get(image_id, -1)

        X = self.X[user_index] if (user_index != -1) else self.avgX
        Y = self.Y[image_index] if (image_index != -1) else self.avgY

        return X @ Y

    def get_recommendations(self, user_id: str, num_recs: int) -> list[str]:
        raise NotImplementedError


def train_mf(train_data: pd.DataFrame) -> MFRecommender:
    graph = LocalGraph()

    recommender = MFRecommender(graph)

    return recommender
