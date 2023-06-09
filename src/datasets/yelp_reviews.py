from pathlib import Path
import json
import pandas as pd

from .datasets import get_raw_data_root, get_clean_data_root


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
                        "photo_id": line["photo_id"],
                        "business_id": line["business_id"],
                        "description": label["description"],
                        "score": label["score"],
                    }
                )

    images_path.parent.mkdir(parents=True, exist_ok=True)
    images_df = pd.DataFrame(data=images_data)
    images_df.to_csv(images_path, index=False)
    return pd.read_csv(images_path)


def load_reviews(reviews_path: Path) -> pd.DataFrame:
    return pd.read_csv(reviews_path)


def gen_reviews(
    yelp_reviews_path: Path, images_df: pd.DataFrame, reviews_path: Path
) -> pd.DataFrame:
    if not yelp_reviews_path.exists():
        raise FileNotFoundError(f"{yelp_reviews_path=} does not exist")

    reviews_data = []
    with open(yelp_reviews_path) as f:
        for line_raw in f:
            line = json.loads(line_raw)
            reviews_data.append(
                {
                    "user_id": line["user_id"],
                    "business_id": line["business_id"],
                    "rating": line["stars"],
                    "timestamp": line["date"],
                }
            )

    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    reviews_df = pd.DataFrame(data=reviews_data)
    reviews_df = reviews_df.merge(
        images_df[["photo_id", "business_id"]].drop_duplicates(), on="business_id"
    )
    weights_df = (
        reviews_df[["user_id", "business_id"]].value_counts().apply(lambda n: 1 / n)
    ).rename("weight")
    reviews_df = reviews_df.merge(
        weights_df, left_on=["user_id", "business_id"], right_index=True
    ).drop(columns="business_id")
    reviews_df.to_csv(reviews_path, index=False)
    return pd.read_csv(reviews_path)


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

        if reviews_path.exists():
            self.reviews = load_reviews(reviews_path)
        elif make_if_not_exists:
            self.reviews = gen_reviews(yelp_reviews_path, self.images, reviews_path)
        else:
            raise FileNotFoundError(
                f"{reviews_path=} does not exist, try setting make_if_not_exists=True"
            )

        # join and add weights?
        # train test splidd?
        # splidd into folds?
