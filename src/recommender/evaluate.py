import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import]

from .recommender import Recommender


def train() -> None:  # TODO: figure out where to put this
    import json
    import logging
    import pandas as pd
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    data = Path("../data")
    yelp_dataset = data / "yelp_dataset"
    if not yelp_dataset.exists():
        raise FileNotFoundError(f"Could not find {yelp_dataset=}")
    reviews = data / "reviews"
    reviews.mkdir(exist_ok=True)

    logging.info("Reading images")
    if (reviews / "images.csv").exists():
        images_df = pd.read_csv(reviews / "images.csv")
    else:
        images_data = {"image_id": [], "tags": []}
        with open(yelp_dataset / "yelp_academic_dataset_business.json") as f:
            for line_raw in f:
                line = json.loads(line_raw)
                images_data["image_id"].append(line.get("business_id", None))
                images_data["tags"].append(line.get("categories", None))
        images_df = pd.DataFrame(images_data)
        images_df.to_csv(reviews / "images.csv", index=None)

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

    raise NotImplementedError


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
