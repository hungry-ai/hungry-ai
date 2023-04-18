import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import]
from pathlib import Path
from sklearn.model_selection import train_test_split

from .recommender import Recommender


def reviews_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        images_data = {"image_id": [], "tags": []}
        with open(yelp_dataset / "yelp_academic_dataset_business.json") as f:
            for line_raw in f:
                line = json.loads(line_raw)
                images_data["image_id"].append(line.get("business_id", None))
                images_data["tags"].append(line.get("categories", None))
        images = pd.DataFrame(images_data)
        images.to_csv(reviews / "images.csv", index=None)

    logging.info("Reading reviews")
    if (reviews / "reviews_train.csv").exists() and (
        reviews / "reviews_test.csv"
    ).exists():
        reviews_train = pd.read_csv(reviews / "reviews_train.csv")
        reviews_test = pd.read_csv(reviews / "reviews_test.csv")
    else:
        reviews_data = {"user_id": [], "image_id": [], "rating": []}
        with open(yelp_dataset / "yelp_academic_dataset_review.json") as f:
            for line_raw in f:
                line = json.loads(line_raw)
                reviews_data["user_id"].append(line.get("user_id", None))
                reviews_data["image_id"].append(line.get("business_id", None))
                reviews_data["rating"].append(line.get("stars", None))
        reviews_df = pd.DataFrame(reviews_data)
        reviews_train, reviews_test = train_test_split(reviews_df, train_size=0.8)
        reviews_train.to_csv(reviews / "reviews_train.csv", index=None)
        reviews_test.to_csv(reviews / "reviews_test.csv", index=None)

    return reviews_train, reviews_test, images


def evaluate_predictions(
    recommender: Recommender, test_set: pd.DataFrame
) -> dict[str, float]:
    # test_set is a csv with columns: user_id, image_id, rating
    # all image_ids should exist in recommender.graph

    rating = test_set["rating"]
    rating_pred = test_set.apply(
        lambda row: recommender.predict_rating(row["user_id"], row["image_id"]), axis=1
    )

    return {
        "mse": np.mean((rating - rating_pred) ** 2),
        "mae": np.mean(np.abs(rating - rating_pred)),
        "rmse": np.sqrt(np.mean((rating - rating_pred) ** 2)),
    }


def evaluate_recommendations(
    recommender: Recommender, test_set: pd.DataFrame
) -> dict[str, float]:
    # test_set is a csv with columns user_id, image_id, rating
    # all image_ids should exist in recommender.graph

    raise NotImplementedError

    # TODO: fill in metrics from research/aggarwal.pdf