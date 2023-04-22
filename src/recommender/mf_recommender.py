import numpy as np
import pandas as pd  # type: ignore[import]
import random
import re
import time

from numba import njit
from typing import Any

from .recommender import Recommender
from ..images import Image, pr_match
from ..reviews import Review
from ..users import User


class MFRecommender(Recommender):
    def __init__(
        self,
        Y: np.ndarray,
        tags: list[str],
        alpha: float,
    ) -> None:
        self.Y = Y
        self.tags = tags
        self.alpha = alpha

        self.d = Y.shape[1]

        self.X = {}  # user_id -> x
        self.x_avg = np.zeros(self.d)

        self.IY = {}  # image_id -> i @ Y
        self.YTIuTIuY = {}  # user_id -> (I_u Y).T (I_u Y) + (alpha nnz / d) I_d
        self.YTIuTru = {}  # user_id -> (I_u Y).T r_u

    def add_user(self, user: User) -> None:
        if user.id in self.X:
            return

        self.X[user.id] = self.x_avg
        self.YTIuTIuY = np.zeros((self.d, self.d))
        self.YTIuTru = np.zeros(self.d)

    def add_image(self, image: Image) -> None:
        tag_prs = [
            (self.Y[tag_index], pr_match(image.url, tag))
            for tag_index, tag in enumerate(self.tags)
        ]
        ys, i = zip(*tag_prs)

        # compute iy
        iy = np.array(ys) @ np.array(i)

        # update iy
        self.IY[image.id] = iy

    def add_review(self, review: Review) -> None:
        user = review.user
        image = review.image

        iy = self.IY[image.id]

        # compute x
        A = self.YTIuTIuY[user.id]
        A += iy.reshape(-1, 1) @ iy.reshape(1, -1)
        A += self.alpha / self.d * np.eye(self.d)
        b = self.YTiTru[user.id]
        b += review.rating * iy

        x = np.linalg.solve(A, b)

        # update x
        self.x_avg += (x - self.X[user.id]) / len(self.x_avg)
        self.X[user.id] = x

    def predict_rating(self, user_id: str, image_id: str) -> float:
        x = self.X[user_id]
        iy = self.IY[image_id]
        return x @ iy

    def get_recommendations(self, user: User, num_recs: int) -> list[str]:
        raise NotImplementedError


def get_user_indices(train_data: pd.DataFrame) -> pd.DataFrame:
    user_ids = train_data["user_id"].drop_duplicates()
    user_indices = user_ids.reset_index(drop=True).reset_index()
    user_indices = user_indices.rename(columns={"index": "user_index"})
    return user_indices


def get_image_indices(train_data: pd.DataFrame) -> pd.DataFrame:
    image_ids = train_data["image_id"].drop_duplicates()
    image_indices = image_ids.reset_index(drop=True).reset_index()
    image_indices = image_indices.rename(columns={"index": "image_index"})
    return image_indices


def get_tags_parsed(images: pd.DataFrame) -> pd.DataFrame:
    splidd_regex = re.compile(",|&|/")
    replace_regex = re.compile(" |\(|\)|'|-")

    def parse_tags(tags_raw: Any) -> list[str]:
        if not isinstance(tags_raw, str):
            return []

        tags = []
        for tag_raw in re.split(splidd_regex, tags_raw):
            tag = re.sub(replace_regex, "", tag_raw.strip().lower())
            if tag:
                tags.append(tag)
        return tags

    tags_parsed = images["tags"].apply(parse_tags).rename("tags_parsed")

    return pd.concat([images, tags_parsed], axis=1)


def get_image_tags(
    images: pd.DataFrame, image_indices: pd.DataFrame
) -> tuple[np.ndarray, list[str], int]:
    tags_parsed = get_tags_parsed(images)

    tags = tags_parsed["tags_parsed"].explode().dropna().unique()
    tag_indices = {tag: tag_index for tag_index, tag in enumerate(tags)}
    k = len(tags)

    def tag_vector(tags: list[str]) -> np.ndarray:
        if not tags:
            return np.zeros((1, k))
        return (sum([np.eye(1, k, tag_indices[tag]) for tag in tags])) / len(tags)

    image_tags_df = image_indices.merge(tags_parsed, on="image_id", how="left")
    np.testing.assert_array_equal(
        image_tags_df["image_index"], np.arange(len(image_indices))
    )
    image_tags = np.concatenate(image_tags_df["tags_parsed"].apply(tag_vector))

    return image_tags, tags, k


def preprocess(
    train_data: pd.DataFrame, images: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, list[str], int, int, int]:
    user_indices = get_user_indices(train_data)
    n = len(user_indices)

    image_indices = get_image_indices(train_data)
    m = len(user_indices)

    image_tags, tags, k = get_image_tags(images, image_indices)

    train_data = train_data.merge(user_indices, on="user_id")
    train_data = train_data.merge(image_indices, on="image_id")

    return train_data, image_tags, tags, n, m, k


def log_elapsed(start: float, end: float, n: int = 0) -> None:
    if n == 0:
        print(f"Elapsed: {end - start:.2f}s")
    else:
        print(f"Elapsed: {end - start:.2f}s, average: {(end - start) / n:.2f}s")


def get_reviews_by_user(
    train_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    print("Precomputing reviews_by_user")

    start = time.time()
    reviews_by_user = (
        train_data.groupby("user_index")
        .apply(
            lambda group: (
                group["image_index"].to_numpy(),
                group["rating"].to_numpy(),
            )
        )
        .to_dict()
    )
    end = time.time()
    log_elapsed(start, end)

    return zip(*[reviews_by_user[u] for u in range(len(reviews_by_user))])


@njit(parallel=True)
def solve_X(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    image_ids_by_user: np.ndarray,
    ratings_by_user: np.ndarray,
    n: int,
    d: int,
    alpha: float,
) -> np.ndarray:
    for u in range(n):
        image_ids = image_ids_by_user[u]
        ratings = ratings_by_user[u]
        A = alpha / (n * d) * np.eye(d)
        b = np.zeros(d)
        for i, r in zip(image_ids, ratings):
            a = I[i] @ Y
            b += r * a
            a = a.reshape(1, -1)
            A += a.T @ a

        X[u] = np.linalg.solve(A, b)


def get_gradient(
    i: int,
    *,
    X: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    image_ids_by_user: np.ndarray,
    ratings_by_user: np.ndarray,
    n: int,
    k: int,
    d: int,
    beta: float,
    sample_size: int = 0,
) -> np.ndarray:
    grad = 2 * beta / (k * d) * Y[i]

    sample_size = sample_size or n
    sample = range(n) if sample_size == n else random.sample(range(n), sample_size)
    for u in sample:
        image_ids = image_ids_by_user[u]
        ratings = ratings_by_user[u]
        io = I[image_ids]
        grad += (
            2
            / (sample_size * len(image_ids))
            * (io[:, i] @ (io @ (Y @ X[u]) - ratings))
            * X[u]
        )

    return grad


def get_loss(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    image_ids_by_user: np.ndarray,
    ratings_by_user: np.ndarray,
    n: int,
    k: int,
    d: int,
    beta: float,
    sample_size: int = 0,
) -> float:
    loss = beta / (k * d) * np.linalg.norm(Y) ** 2

    sample_size = sample_size or n
    sample = range(n) if sample_size == n else random.sample(range(n), sample_size)
    for u in sample:
        image_ids = image_ids_by_user[u]
        ratings = ratings_by_user[u]
        if len(image_ids) > 0:
            loss += (
                1
                / (sample_size * len(image_ids))
                * sum((ratings - I[image_ids] @ (Y @ X[u])) ** 2)
            )

    return loss


def solve_Y(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    image_ids_by_user: np.ndarray,
    ratings_by_user: np.ndarray,
    n: int,
    k: int,
    d: int,
    beta: float,
    epochs: int,
    learning_rate: int,
    batch_size: int,
) -> np.ndarray:
    print("Computing Y")

    start = time.time()
    for sgd_epoch in range(epochs):
        print(f"{sgd_epoch=}")

        for i in range(k):
            if i % 100 == 0:
                print(i)

            gradient = get_gradient(
                i,
                X=X,
                Y=Y,
                I=I,
                image_ids_by_user=image_ids_by_user,
                ratings_by_user=ratings_by_user,
                n=n,
                k=k,
                d=d,
                beta=beta,
                sample_size=batch_size,
            )
            Y[i] -= learning_rate / (sgd_epoch + 1) * gradient
        end = time.time()
        log_elapsed(start, end)

        loss = get_loss(
            X=X,
            Y=Y,
            I=I,
            image_ids_by_user=image_ids_by_user,
            ratings_by_user=ratings_by_user,
            n=n,
            k=k,
            d=d,
            beta=beta,
            sample_size=batch_size,
        )
        end = time.time()
        print(f"{loss=}")
        log_elapsed(start, end, sgd_epoch + 1)


def train_mf(
    train_data: pd.DataFrame,
    images: pd.DataFrame,
    *,
    d: int,
    alpha: float,
    beta: float,
    als_epochs: int,
    sgd_learning_rate: int,
    sgd_epochs: int,
    sgd_batch_size: int,
) -> MFRecommender:
    train_data, I, tags, n, m, k = preprocess(train_data, images)

    image_ids_by_user, ratings_by_user = get_reviews_by_user(train_data)

    X = np.random.normal(size=(n, d))
    Y = np.random.normal(size=(k, d))

    for als_epoch in range(als_epochs):
        print("Computing X")

        start = time.time()
        solve_X(
            X=X,
            Y=Y,
            I=I,
            image_ids_by_user=image_ids_by_user,
            ratings_by_user=ratings_by_user,
            n=n,
            d=d,
            alpha=alpha,
        )
        end = time.time()
        log_elapsed(start, end)

        solve_Y(
            X=X,
            Y=Y,
            I=I,
            image_ids_by_user=image_ids_by_user,
            ratings_by_user=ratings_by_user,
            n=n,
            k=k,
            d=d,
            beta=beta,
            epochs=sgd_epochs,
            learning_rate=sgd_learning_rate,
            batch_size=sgd_batch_size,
        )

    return MFRecommender(Y, tags, alpha)
