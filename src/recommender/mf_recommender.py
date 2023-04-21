import heapq
import pandas as pd  # type: ignore[import]
import numpy as np

from .recommender import Recommender
from ..tags import WordEmbedding, generate_tags_graph


class MFRecommender(Recommender):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        user_indices: dict[str, int],
        image_indices: dict[str, int],
    ) -> None:
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


def train_mf(
    train_data: pd.DataFrame, K: int, alpha: float, iterations: int
) -> MFRecommender:
    user_ids = train_data["user_id"].drop_duplicates()
    user_indices = user_ids.reset_index(drop=True).reset_index()
    user_indices = user_indices.rename(columns={"index": "user_index"})

    image_ids = train_data["image_id"].drop_duplicates()
    image_indices = image_ids.reset_index(drop=True).reset_index()
    image_indices = image_indices.rename(columns={"index": "image_index"})

    train_data = train_data.merge(user_indices, on="user_id")
    train_data = train_data.merge(image_indices, on="image_id")

    N = train_data["user_index"].max() + 1
    M = train_data["image_index"].max() + 1

    X = np.ones((N, K))
    Y = np.ones((M, K))

    print("Precomputing 1")
    user_to_image = (
        train_data.groupby("user_index")["image_ndex"]
        .apply(lambda x: x.values)
        .to_dict()
    )

    print("Precomputing 2")
    user_to_image_rating = (
        train_data.groupby("user_index")["rating"].apply(lambda x: x.values).to_dict()
    )

    print("Precomputing 3")
    image_to_user = (
        train_data.groupby("image_index")["user_index"]
        .apply(lambda x: x.values)
        .to_dict()
    )

    print("Precomputing 4")
    image_to_user_rating = (
        train_data.groupby("image_index")["rating"].apply(lambda x: x.values).to_dict()
    )

    for it in range(iterations):  # Change to convergence
        print("At iteration", it)
        for u in range(N):
            if u % 10000 == 0:
                print("User", u)
            YITYI = np.zeros((K, K))

            for i in user_to_image[u]:
                YI = Y[i].reshape(1, -1)
                YITYI += YI.T @ YI
            YITYI += alpha * np.identity(K)
            YITYI_inv = np.linalg.inv(YITYI)

            RYT = Y[user_to_image[u]].T @ (user_to_image_rating[u]).reshape(1, -1).T

            X[u] = (YITYI_inv @ RYT).reshape(-1)

        for m in range(M):
            if m % 10000 == 0:
                print("Image", m)
            XITXI = np.zeros((K, K))

            for i in image_to_user[m]:
                XI = X[i].reshape(1, -1)
                XITXI += XI.T @ XI
            XITXI += alpha * np.identity(K)
            XITXI_inv = np.linalg.inv(XITXI)

            RXT = X[image_to_user[m]].T @ (image_to_user_rating[m]).reshape(1, -1).T

            Y[m] = (XITXI_inv @ RXT).reshape(-1)

    recommender = MFRecommender(
        X=X, Y=Y, user_indices=user_indices, image_indices=image_indices
    )

    return recommender
