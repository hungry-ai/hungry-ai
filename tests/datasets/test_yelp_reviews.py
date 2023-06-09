from src.datasets import YelpReviews

import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture(scope="function")
def data_root(root: Path) -> Path:
    return root / "data"


@pytest.fixture(scope="function")
def raw_data_root(data_root: Path) -> Path:
    return data_root / "raw"


@pytest.fixture(scope="function")
def clean_data_root(data_root: Path) -> Path:
    return data_root / "clean"


@pytest.fixture(scope="function")
def image_tags_raw() -> str:
    return """{"photo_id": "zsvj7vloL4L5jhYyPIuVwg", "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw", "labels": [{"mid": "/m/06z37_", "description": "Picture frame", "score": 0.8601544499397278, "topicality": 0.8601544499397278}, {"mid": "/m/0ch399z", "description": "Drinking establishment", "score": 0.8142252564430237, "topicality": 0.8142252564430237}, {"mid": "/m/0h8nvcj", "description": "Barware", "score": 0.8120356202125549, "topicality": 0.8120356202125549}, {"mid": "/m/06pg22", "description": "Snapshot", "score": 0.7428473830223083, "topicality": 0.7428473830223083}, {"mid": "/m/0jjw", "description": "Art", "score": 0.7309282422065735, "topicality": 0.7309282422065735}, {"mid": "/m/01c648", "description": "Laptop", "score": 0.7084137797355652, "topicality": 0.7084137797355652}, {"mid": "/m/0271t", "description": "Drink", "score": 0.6847642660140991, "topicality": 0.6847642660140991}, {"mid": "/m/06ht1", "description": "Room", "score": 0.6833332777023315, "topicality": 0.6833332777023315}, {"mid": "/m/081pkj", "description": "Event", "score": 0.6618310809135437, "topicality": 0.6618310809135437}, {"mid": "/m/0dkw5", "description": "Machine", "score": 0.6545807719230652, "topicality": 0.6545807719230652}]}
{"photo_id": "7R6g6VwRIhU3hxB1Huw9Kg", "business_id": "7ATYjTIgM3jUlt4UM3IypQ", "labels": [{"mid": "/m/0ch399z", "description": "Drinking establishment", "score": 0.9627104997634888, "topicality": 0.9627104997634888}, {"mid": "/m/0h8nvcj", "description": "Barware", "score": 0.8916071653366089, "topicality": 0.8916071653366089}, {"mid": "/m/012mj", "description": "Alcoholic beverage", "score": 0.8583782315254211, "topicality": 0.8583782315254211}, {"mid": "/m/0271t", "description": "Drink", "score": 0.84905606508255, "topicality": 0.84905606508255}, {"mid": "/m/04dr76w", "description": "Bottle", "score": 0.8332307934761047, "topicality": 0.8332307934761047}, {"mid": "/m/06z37_", "description": "Picture frame", "score": 0.8282206654548645, "topicality": 0.8282206654548645}, {"mid": "/m/02k1gj", "description": "Beer tap", "score": 0.7983258962631226, "topicality": 0.7983258962631226}, {"mid": "/m/0lqk", "description": "Alcohol", "score": 0.7709624171257019, "topicality": 0.7709624171257019}, {"mid": "/m/04shl0", "description": "Distilled beverage", "score": 0.742070734500885, "topicality": 0.742070734500885}, {"mid": "/m/039jq", "description": "Glass", "score": 0.7310327291488647, "topicality": 0.7310327291488647}]}
{"photo_id": "9Icdczc6d-YA1ubWzbmPRA", "business_id": "7ATYjTIgM3jUlt4UM3IypQ", "labels": [{"mid": "/m/0ch399z", "description": "Drinking establishment", "score": 0.9428887963294983, "topicality": 0.9428887963294983}, {"mid": "/m/0h8nvcj", "description": "Barware", "score": 0.8853147625923157, "topicality": 0.8853147625923157}, {"mid": "/m/06z37_", "description": "Picture frame", "score": 0.8851861357688904, "topicality": 0.8851861357688904}, {"mid": "/m/0271t", "description": "Drink", "score": 0.8072800040245056, "topicality": 0.8072800040245056}, {"mid": "/m/04bcr3", "description": "Table", "score": 0.7969378232955933, "topicality": 0.7969378232955933}, {"mid": "/m/01mzpv", "description": "Chair", "score": 0.7687520980834961, "topicality": 0.7687520980834961}, {"mid": "/m/012mj", "description": "Alcoholic beverage", "score": 0.7631221413612366, "topicality": 0.7631221413612366}, {"mid": "/m/0cgh4", "description": "Building", "score": 0.7327222228050232, "topicality": 0.7327222228050232}, {"mid": "/m/016m2d", "description": "Skull", "score": 0.7116127610206604, "topicality": 0.7116127610206604}, {"mid": "/m/01b92", "description": "Bone", "score": 0.6910802125930786, "topicality": 0.6910802125930786}]}"""


@pytest.fixture(scope="function")
def yelp_reviews_raw() -> str:
    return """{"review_id":"KU_O5udG6zpxOg-VcAEodg","user_id":"mh_-eMZ6K5RLWhZyISBhwA","business_id":"XQfwVwDr-v0ZS3_CbbE5Xw","stars":3.0,"useful":0,"funny":0,"cool":0,"text":"If you decide to eat here, just be aware it is going to take about 2 hours from beginning to end. We have tried it multiple times, because I want to like it! I have been to it's other locations in NJ and never had a bad experience. \\n\\nThe food is good, but it takes a very long time to come out. The waitstaff is very young, but usually pleasant. We have just had too many experiences where we spent way too long waiting. We usually opt for another diner or restaurant on the weekends, in order to be done quicker.","date":"2018-07-07 22:09:11"}
{"review_id":"BiTunyQ73aT9WBnpR9DZGw","user_id":"OyoGAe7OKpv6SyGZT5g77Q","business_id":"7ATYjTIgM3jUlt4UM3IypQ","stars":5.0,"useful":1,"funny":0,"cool":1,"text":"I've taken a lot of spin classes over the years, and nothing compares to the classes at Body Cycle. From the nice, clean space and amazing bikes, to the welcoming and motivating instructors, every class is a top notch work out.\\n\\nFor anyone who struggles to fit workouts in, the online scheduling system makes it easy to plan ahead (and there's no need to line up way in advanced like many gyms make you do).\\n\\nThere is no way I can write this review without giving Russell, the owner of Body Cycle, a shout out. Russell's passion for fitness and cycling is so evident, as is his desire for all of his clients to succeed. He is always dropping in to classes to check in/provide encouragement, and is open to ideas and recommendations from anyone. Russell always wears a smile on his face, even when he's kicking your butt in class!","date":"2012-01-03 15:28:18"}"""


def test_reviews_dataset(
    raw_data_root: Path,
    clean_data_root,
    image_tags_raw: str,
    yelp_reviews_raw: str,
) -> None:
    image_tags_path = raw_data_root / "image_tags.json"
    yelp_reviews_path = (
        raw_data_root / "yelp_dataset" / "yelp_academic_dataset_review.json"
    )

    with pytest.raises(FileNotFoundError):
        YelpReviews(
            images_path=clean_data_root / "yelp_images_1.csv",
            reviews_path=clean_data_root / "yelp_reviews_1.csv",
            image_tags_path=image_tags_path,
            yelp_reviews_path=yelp_reviews_path,
            make_if_not_exists=True,
        )

    yelp_reviews_path.parent.mkdir(exist_ok=True, parents=True)
    with open(yelp_reviews_path, "w+") as f:
        f.write(yelp_reviews_raw)

    with pytest.raises(FileNotFoundError):
        YelpReviews(
            images_path=clean_data_root / "yelp_images_2.csv",
            reviews_path=clean_data_root / "yelp_reviews_2.csv",
            image_tags_path=image_tags_path,
            yelp_reviews_path=yelp_reviews_path,
            make_if_not_exists=True,
        )

    with pytest.raises(FileNotFoundError):
        YelpReviews(
            images_path=clean_data_root / "yelp_images_3.csv",
            reviews_path=clean_data_root / "yelp_reviews_3.csv",
            image_tags_path=image_tags_path,
            yelp_reviews_path=yelp_reviews_path,
            make_if_not_exists=True,
        )

    image_tags_path.parent.mkdir(exist_ok=True, parents=True)
    with open(image_tags_path, "w+") as f:
        f.write(image_tags_raw)

    yelp_reviews_1 = YelpReviews(
        images_path=clean_data_root / "yelp_images.csv",
        reviews_path=clean_data_root / "yelp_reviews.csv",
        image_tags_path=image_tags_path,
        yelp_reviews_path=yelp_reviews_path,
        make_if_not_exists=True,
    )

    assert isinstance(yelp_reviews_1.images, pd.DataFrame)
    assert list(yelp_reviews_1.images.columns) == [
        "photo_id",
        "business_id",
        "description",
        "score",
    ]
    assert len(yelp_reviews_1.images) == 30

    assert isinstance(yelp_reviews_1.reviews, pd.DataFrame)
    assert list(yelp_reviews_1.reviews.columns) == [
        "user_id",
        "rating",
        "timestamp",
        "photo_id",
        "weight",
    ]
    assert len(yelp_reviews_1.reviews) == 3

    with pytest.raises(FileNotFoundError):
        YelpReviews(
            images_path=clean_data_root / "yelp_images_4.csv",
            reviews_path=clean_data_root / "yelp_reviews_4.csv",
        )

    with pytest.raises(FileNotFoundError):
        YelpReviews(
            images_path=clean_data_root / "yelp_images.csv",
            reviews_path=clean_data_root / "yelp_reviews_5.csv",
        )

    yelp_reviews_2 = YelpReviews(
        images_path=clean_data_root / "yelp_images.csv",
        reviews_path=clean_data_root / "yelp_reviews.csv",
    )

    assert yelp_reviews_2.images.equals(yelp_reviews_1.images)
    assert yelp_reviews_2.reviews.equals(yelp_reviews_1.reviews)
