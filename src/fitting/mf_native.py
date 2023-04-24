import random
import re
import time
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import]
from numba import njit, prange  # type: ignore[import]

from ..recommender import MFRecommender


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
    replace_regex = re.compile(" |\\(|\\)|'|-")

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

        return np.average(
            np.array([np.eye(1, k, tag_indices[tag]).reshape(-1) for tag in tags]),
            axis=0,
        )

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


def get_reviews_by_user(
    train_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sorted_data = train_data.sort_values(["user_index", "image_index"]).reset_index(
        drop=True
    )
    image_indices = sorted_data["image_index"].to_numpy()
    ratings = sorted_data["rating"].to_numpy()
    start_indices = (
        sorted_data.reset_index().groupby("user_index")["index"].first().to_numpy()
    )
    end_indices = (
        sorted_data.reset_index().groupby("user_index")["index"].last().to_numpy() + 1
    )
    return image_indices, ratings, start_indices, end_indices


@njit()
def get_loss(
    X: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    image_indices: np.ndarray,
    ratings: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    n: int,
    k: int,
    d: int,
    alpha: float,
    beta: float,
) -> float:
    loss = 0.0
    penalty_x = 0.0
    penalty_y = 0.0

    for u in prange(n):
        for i in prange(start_indices[u], end_indices[u]):
            loss += (ratings[i] - I[image_indices[i]] @ Y @ X[u]) ** 2

        penalty_x += np.sum(X[u] ** 2)

    for t in range(k):
        penalty_y += np.sum(Y[t] ** 2)

    return loss / n + alpha / (n * d) * penalty_x + beta / (k * d) * penalty_y


@njit()
def update_X(
    X: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    image_indices: np.ndarray,
    ratings: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    n: int,
    d: int,
    alpha: float,
) -> None:
    for u in prange(n):
        A = (alpha * (end_indices[u] - start_indices[u]) / d) * np.eye(d)
        b = np.zeros(d)

        for i in prange(start_indices[u], end_indices[u]):
            iy = I[image_indices[i]] @ Y

            A += iy.reshape(-1, 1) @ iy.reshape(1, -1)
            b += ratings[i] * iy

        X[u] = np.linalg.solve(A, b)


def update_gradient(
    gradient: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    image_indices: np.ndarray,
    ratings: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    n: int,
    k: int,
    d: int,
    beta: float,
    batch_size: int,
) -> None:
    sample = random.sample(range(n), batch_size)

    start = time.time()
    for u_index in prange(batch_size):
        u = sample[u_index]
        if u_index % 10000 == 0 and u_index > 0:
            end = time.time()
            print(
                f"{u_index} users processed, elapsed: {end - start:.2f}s, average: {(end-start)/u*10000:.2f}s"
            )

        a = np.zeros(k)

        for i in prange(start_indices[u], end_indices[u]):
            image_index = image_indices[i]
            rating = ratings[i]

            a += (rating - I[image_indices[i]] @ Y @ X[u]) * I[image_index]

        gradient += (
            a.reshape(-1, 1) @ X[u].reshape(1, -1) / (end_indices[u] - start_indices[u])
        )

    gradient *= -2 / batch_size
    gradient += (2 * beta) / (k * d) * Y


def update_Y(
    X: np.ndarray,
    Y: np.ndarray,
    I: np.ndarray,
    image_indices: np.ndarray,
    ratings: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    n: int,
    k: int,
    d: int,
    beta: float,
    learning_rate: float,
    max_epochs: int,
    batch_size: int,
) -> None:
    delta_Y = np.zeros((k, d))

    start = time.time()
    for epoch in range(1, max_epochs + 1):
        gradient = np.zeros((k, d))
        update_gradient(
            gradient,
            X=X,
            Y=Y,
            I=I,
            image_indices=image_indices,
            ratings=ratings,
            start_indices=start_indices,
            end_indices=end_indices,
            n=n,
            k=k,
            d=d,
            beta=beta,
            batch_size=batch_size,
        )

        Y -= learning_rate * gradient
        delta_Y += gradient

        if epoch % 100000 == 0 and epoch > 0:
            end = time.time()
            delta_Y_size = (delta_Y**2).sum()
            Y_size = (Y**2).sum()
            percent_change = delta_Y_size / Y_size
            print(
                f"{epoch=}, elapsed: {end-start:.2f}s, average: {(end-start)/epoch*100000:.2f}s, {delta_Y_size=:.2e}, {Y_size=:.2e}, {percent_change=:.2e}"
            )
            delta_Y = np.zeros((k, d))


def train_mf_native(
    train_data: pd.DataFrame,
    images: pd.DataFrame,
    *,
    d: int,
    alpha: float,
    beta: float,
    max_als_epochs: int,  # TODO: add convergence condition
    sgd_learning_rate: float,
    max_sgd_epochs: int,  # TODO: add convergence condition
    sgd_batch_size: int,
) -> MFRecommender:
    train_data, I, tags, n, m, k = preprocess(train_data, images)

    image_indices, ratings, start_indices, end_indices = get_reviews_by_user(train_data)

    X = np.random.normal(size=(n, d))
    Y = np.random.normal(size=(k, d))

    for als_epoch in range(max_als_epochs):
        print(f"{als_epoch=}")

        print("Computing X")
        start = time.time()
        update_X(
            X=X,
            Y=Y,
            I=I,
            image_indices=image_indices,
            ratings=ratings,
            start_indices=start_indices,
            end_indices=end_indices,
            n=n,
            d=d,
            alpha=alpha,
        )
        end = time.time()
        print(f"{end - start:.2f}s")

        print("Computing Y")
        start = time.time()
        update_Y(
            X=X,
            Y=Y,
            I=I,
            image_indices=image_indices,
            ratings=ratings,
            start_indices=start_indices,
            end_indices=end_indices,
            n=n,
            k=k,
            d=d,
            beta=beta,
            learning_rate=sgd_learning_rate,
            max_epochs=max_sgd_epochs,
            batch_size=sgd_batch_size,
        )
        end = time.time()
        print(f"{end - start:.2f}s")

        print("Computing loss")
        start = time.time()
        loss = get_loss(
            X=X,
            Y=Y,
            I=I,
            image_indices=image_indices,
            ratings=ratings,
            start_indices=start_indices,
            end_indices=end_indices,
            n=n,
            k=k,
            d=d,
            alpha=alpha,
            beta=beta,
        )
        print(f"{loss=}")
        end = time.time()
        print(f"{end - start:.2f}s")

    return MFRecommender(Y, tags, alpha)
