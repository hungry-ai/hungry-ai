import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import]
from sklearn.model_selection import train_test_split  # type: ignore[import]

from ..recommender import Recommender


def reviews_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = Path("../data")
    yelp_dataset = data / "yelp_dataset"
    if not yelp_dataset.exists():
        raise FileNotFoundError(f"could not find {yelp_dataset=}")
    reviews = data / "reviews"
    reviews.mkdir(exist_ok=True)

    logging.info("Reading images")
    if (reviews / "images.csv").exists():
        images = pd.read_csv(reviews / "images.csv")
    else:
        images_data: dict[str, list[Any]] = {"image_id": [], "tags": []}
        with open(yelp_dataset / "yelp_academic_dataset_business.json") as f:
            for line_raw in f:
                line = json.loads(line_raw)
                images_data["image_id"].append(line.get("business_id", None))
                images_data["tags"].append(line.get("categories", None) or "")
        images = pd.DataFrame(images_data)
        images.to_csv(reviews / "images.csv", index=None)

    logging.info("Reading reviews")
    if (
        (reviews / "reviews_train.csv").exists()
        and (reviews / "reviews_validation.csv").exists()
        and (reviews / "reviews_test.csv").exists()
    ):
        reviews_train = pd.read_csv(reviews / "reviews_train.csv")
        reviews_validation = pd.read_csv(reviews / "reviews_validation.csv")
        reviews_test = pd.read_csv(reviews / "reviews_test.csv")
    else:
        reviews_data: dict[str, list[Any]] = {
            "user_id": [],
            "image_id": [],
            "rating": [],
        }
        with open(yelp_dataset / "yelp_academic_dataset_review.json") as f:
            for line_raw in f:
                line = json.loads(line_raw)
                reviews_data["user_id"].append(line.get("user_id", None))
                reviews_data["image_id"].append(line.get("business_id", None))
                reviews_data["rating"].append(line.get("stars", None))
        reviews_df = pd.DataFrame(reviews_data)
        reviews_train, reviews_test = train_test_split(
            reviews_df, train_size=0.8, random_state=42069
        )
        reviews_validation, reviews_test = train_test_split(
            reviews_test, train_size=0.5, random_state=42069
        )
        reviews_train.to_csv(reviews / "reviews_train.csv", index=None)
        reviews_test.to_csv(reviews / "reviews_validation.csv", index=None)
        reviews_test.to_csv(reviews / "reviews_test.csv", index=None)

    return reviews_train, reviews_validation, reviews_test, images


def evaluate_predictions(
    recommender: Recommender, test_set: pd.DataFrame
) -> dict[str, float]:
    # test_set is a csv with columns: user_id, image_id, rating
    # all image_ids should exist in recommender.graph

    raise NotImplementedError


def evaluate_recommendations(
    recommender: Recommender, test_set: pd.DataFrame
) -> dict[str, float]:
    # test_set is a csv with columns user_id, image_id, rating
    # all image_ids should exist in recommender.graph

    raise NotImplementedError

    # TODO: fill in metrics from research/aggarwal.pdf
