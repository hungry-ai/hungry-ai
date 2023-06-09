from pathlib import Path
import json
import numpy as np
import pandas as pd
import warnings

from .datasets import get_raw_data_root, get_clean_data_root


def add_index(df: pd.DataFrame, column: str) -> pd.DataFrame:
    index = (
        df[f"{column}_id"]
        .drop_duplicates()
        .reset_index()
        .rename(columns={"index": f"{column}_index"})
    )
    return df.merge(index, on=f"{column}_id")


def load_images(images_path: Path) -> pd.DataFrame:
    return pd.read_csv(images_path)


def gen_images(image_tags_path: Path, images_path: Path) -> pd.DataFrame:
    if not image_tags_path.exists():
        raise FileNotFoundError(f"{image_tags_path=} does not exist")

    images_data = []
    with open(image_tags_path) as f:
        for line_raw in f:
            line = json.loads(line_raw)
            for label in line["labels"]:
                images_data.append(
                    {
                        "image_id": line["photo_id"],
                        "restaurant_id": line["business_id"],
                        "tag_id": label["description"],
                        "weight": label["score"],
                    }
                )

    images_path.parent.mkdir(parents=True, exist_ok=True)
    images_df = pd.DataFrame(data=images_data)

    images_df = add_index(images_df, "image")
    images_df = add_index(images_df, "tag")

    images_df.to_csv(images_path, index=False)
    return load_images(images_path)


def train_test_splidd(
    reviews: pd.DataFrame, train_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_weight_cumsum = (
        reviews.groupby("user_id")["weight"]
        .sum()
        .sample(frac=1, random_state=42069)
        .cumsum()
    )

    total_weight = user_weight_cumsum.iloc[-1]

    train_users = user_weight_cumsum[user_weight_cumsum <= train_size * total_weight]
    reviews_train = reviews.merge(
        train_users, on="user_id", suffixes=("", "_bad")
    ).drop(columns="weight_bad")

    if reviews_train.empty:
        warnings.warn("train_test_splidd produced an empty training sample")

    test_users = user_weight_cumsum[user_weight_cumsum > train_size * total_weight]
    reviews_test = reviews.merge(test_users, on="user_id", suffixes=("", "_bad")).drop(
        columns="weight_bad"
    )

    if reviews_test.empty:
        warnings.warn("train_test_splidd produced an empty testing sample")

    return reviews_train, reviews_test


class Fold:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame, i: int) -> None:
        self.train = train
        self.test = test
        self.i = i


def k_folds_splidd(reviews: pd.DataFrame, n_folds: int) -> list[Fold]:
    user_weight_cumsum = (
        reviews.groupby("user_id")["weight"]
        .sum()
        .sample(frac=1, random_state=42069)
        .cumsum()
    )

    total_weight = user_weight_cumsum.iloc[-1]

    folds = []
    for i in range(n_folds):
        is_test_user = (i == 0 or user_weight_cumsum > i / n_folds * total_weight) & (
            i == n_folds - 1 or user_weight_cumsum <= (i + 1) / n_folds * total_weight
        )

        train_users = user_weight_cumsum[~is_test_user]
        train = reviews.merge(train_users, on="user_id", suffixes=("", "_bad")).drop(
            columns="weight_bad"
        )

        if train.empty:
            warnings.warn(
                f"k_folds_splidd produced an empty training sample on fold {i}"
            )

        test_users = user_weight_cumsum[is_test_user]
        test = reviews.merge(test_users, on="user_id", suffixes=("", "_bad")).drop(
            columns="weight_bad"
        )

        if test.empty:
            warnings.warn(
                f"k_folds_splidd produced an empty testing sample on fold {i}"
            )

        folds.append(Fold(train, test, i))

    return folds


def load_reviews(
    reviews_train_path: Path,
    reviews_test_path: Path,
    reviews_fold_paths: list[tuple[Path, Path]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[Fold]]:
    return (
        pd.read_csv(reviews_train_path),
        pd.read_csv(reviews_test_path),
        [
            Fold(pd.read_csv(fold_train_path), pd.read_csv(fold_test_path), i)
            for i, (fold_train_path, fold_test_path) in enumerate(reviews_fold_paths)
        ],
    )


def gen_reviews(
    yelp_reviews_path: Path,
    images_df: pd.DataFrame,
    reviews_train_path: Path,
    reviews_test_path: Path,
    reviews_fold_paths: list[tuple[Path, Path]],
    train_size: float,
) -> tuple[list[tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    if not yelp_reviews_path.exists():
        raise FileNotFoundError(f"{yelp_reviews_path=} does not exist")

    reviews_data = []
    with open(yelp_reviews_path) as f:
        for line_raw in f:
            line = json.loads(line_raw)
            reviews_data.append(
                {
                    "user_id": line["user_id"],
                    "restaurant_id": line["business_id"],
                    "rating": line["stars"],
                    "timestamp": line["date"],
                }
            )

    reviews_df = pd.DataFrame(data=reviews_data)
    reviews_df = reviews_df.merge(
        images_df[["image_id", "image_index", "restaurant_id"]].drop_duplicates(),
        on="restaurant_id",
    )

    weights_df = (
        reviews_df[["user_id", "restaurant_id"]].value_counts().apply(lambda n: 1 / n)
    ).rename("weight")
    reviews_df = reviews_df.merge(
        weights_df, left_on=["user_id", "restaurant_id"], right_index=True
    ).drop(columns="restaurant_id")

    train, test = train_test_splidd(reviews_df, train_size=train_size)
    folds = k_folds_splidd(train, n_folds=len(reviews_fold_paths))

    train = add_index(train, "user")
    test = add_index(test, "user")
    for fold in folds:
        fold.train = add_index(fold.train, "user")
        fold.test = add_index(fold.test, "user")

    reviews_train_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(reviews_train_path, index=False)

    reviews_test_path.parent.mkdir(parents=True, exist_ok=True)
    test.to_csv(reviews_test_path, index=False)

    for fold, (fold_train_path, fold_test_path) in zip(folds, reviews_fold_paths):
        fold_train_path.parent.mkdir(parents=True, exist_ok=True)
        fold.train.to_csv(fold_train_path, index=False)

        fold_test_path.parent.mkdir(parents=True, exist_ok=True)
        fold.test.to_csv(fold_test_path, index=False)

    return load_reviews(reviews_train_path, reviews_test_path, reviews_fold_paths)


def get_reviews_paths(
    reviews_path: Path, n_folds: int
) -> tuple[Path, Path, list[tuple[Path, Path]]]:
    prefix, suffix = reviews_path.name.split(".", 1)

    reviews_train_path = reviews_path.parent / f"{prefix}_train.{suffix}"
    reviews_test_path = reviews_path.parent / f"{prefix}_test.{suffix}"

    reviews_fold_paths = [
        (
            reviews_path.parent / f"{prefix}_train_{n_folds}_{i}.{suffix}",
            reviews_path.parent / f"{prefix}_test_{n_folds}_{i}.{suffix}",
        )
        for i in range(n_folds)
    ]

    return reviews_train_path, reviews_test_path, reviews_fold_paths


class YelpReviews:
    def __init__(
        self,
        images_path: Path = get_clean_data_root() / "yelp_images.csv",
        reviews_path: Path = get_clean_data_root() / "yelp_reviews.csv",
        image_tags_path: Path = get_raw_data_root()
        / "yelp_dataset"
        / "yelp_academic_dataset_review.json",
        yelp_reviews_path: Path = get_raw_data_root()
        / "yelp_dataset"
        / "yelp_academic_dataset_review.json",
        train_size: float = 0.8,
        n_folds: int = 5,
        make_if_not_exists: bool = False,
    ) -> None:
        if images_path.exists():
            self.images = load_images(images_path)
        elif make_if_not_exists:
            self.images = gen_images(image_tags_path, images_path)
        else:
            raise FileNotFoundError(
                f"{images_path=} does not exist, try setting make_if_not_exists=True"
            )

        reviews_train_path, reviews_test_path, reviews_fold_paths = get_reviews_paths(
            reviews_path, n_folds
        )
        nonexistent_reviews_paths = []

        if not reviews_train_path.exists():
            nonexistent_reviews_paths.append(reviews_test_path)

        if not reviews_test_path.exists():
            nonexistent_reviews_paths.append(reviews_test_path)

        nonexistent_reviews_paths = [
            path
            for fold_paths in reviews_fold_paths
            for path in fold_paths
            if not path.exists()
        ]

        if len(nonexistent_reviews_paths) == 0:
            self.reviews_train, self.reviews_test, self.reviews_folds = load_reviews(
                reviews_train_path, reviews_test_path, reviews_fold_paths
            )
        elif make_if_not_exists:
            self.reviews_train, self.reviews_test, self.reviews_folds = gen_reviews(
                yelp_reviews_path,
                self.images,
                reviews_train_path,
                reviews_test_path,
                reviews_fold_paths,
                train_size=train_size,
            )
        else:
            raise FileNotFoundError(
                f"{nonexistent_reviews_paths=} do not exist, try setting make_if_not_exists=True"
            )
