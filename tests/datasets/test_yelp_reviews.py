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
{"photo_id": "7R6g6VwRIhU3hxB1Huw9Kg", "business_id": "XQfwVwDr-v0ZS3_CbbE5Xw", "labels": [{"mid": "/m/0ch399z", "description": "Drinking establishment", "score": 0.9627104997634888, "topicality": 0.9627104997634888}, {"mid": "/m/0h8nvcj", "description": "Barware", "score": 0.8916071653366089, "topicality": 0.8916071653366089}, {"mid": "/m/012mj", "description": "Alcoholic beverage", "score": 0.8583782315254211, "topicality": 0.8583782315254211}, {"mid": "/m/0271t", "description": "Drink", "score": 0.84905606508255, "topicality": 0.84905606508255}, {"mid": "/m/04dr76w", "description": "Bottle", "score": 0.8332307934761047, "topicality": 0.8332307934761047}, {"mid": "/m/06z37_", "description": "Picture frame", "score": 0.8282206654548645, "topicality": 0.8282206654548645}, {"mid": "/m/02k1gj", "description": "Beer tap", "score": 0.7983258962631226, "topicality": 0.7983258962631226}, {"mid": "/m/0lqk", "description": "Alcohol", "score": 0.7709624171257019, "topicality": 0.7709624171257019}, {"mid": "/m/04shl0", "description": "Distilled beverage", "score": 0.742070734500885, "topicality": 0.742070734500885}, {"mid": "/m/039jq", "description": "Glass", "score": 0.7310327291488647, "topicality": 0.7310327291488647}]}
{"photo_id": "9Icdczc6d-YA1ubWzbmPRA", "business_id": "7ATYjTIgM3jUlt4UM3IypQ", "labels": [{"mid": "/m/0ch399z", "description": "Drinking establishment", "score": 0.9428887963294983, "topicality": 0.9428887963294983}, {"mid": "/m/0h8nvcj", "description": "Barware", "score": 0.8853147625923157, "topicality": 0.8853147625923157}, {"mid": "/m/06z37_", "description": "Picture frame", "score": 0.8851861357688904, "topicality": 0.8851861357688904}, {"mid": "/m/0271t", "description": "Drink", "score": 0.8072800040245056, "topicality": 0.8072800040245056}, {"mid": "/m/04bcr3", "description": "Table", "score": 0.7969378232955933, "topicality": 0.7969378232955933}, {"mid": "/m/01mzpv", "description": "Chair", "score": 0.7687520980834961, "topicality": 0.7687520980834961}, {"mid": "/m/012mj", "description": "Alcoholic beverage", "score": 0.7631221413612366, "topicality": 0.7631221413612366}, {"mid": "/m/0cgh4", "description": "Building", "score": 0.7327222228050232, "topicality": 0.7327222228050232}, {"mid": "/m/016m2d", "description": "Skull", "score": 0.7116127610206604, "topicality": 0.7116127610206604}, {"mid": "/m/01b92", "description": "Bone", "score": 0.6910802125930786, "topicality": 0.6910802125930786}]}
{"photo_id": "7JxYZJxwVLzJ0VBlgy9Syw", "business_id": "YjUWPpI6HXG530lwP-fb2A", "labels": [{"mid": "/m/0c_jw", "description": "Furniture", "score": 0.9450739026069641, "topicality": 0.9450739026069641}, {"mid": "/m/05wrt", "description": "Property", "score": 0.9430437088012695, "topicality": 0.9430437088012695}, {"mid": "/m/02crq1", "description": "Couch", "score": 0.9363430142402649, "topicality": 0.9363430142402649}, {"mid": "/m/04bcr3", "description": "Table", "score": 0.9299288392066956, "topicality": 0.9299288392066956}, {"mid": "/m/02rfdq", "description": "Interior design", "score": 0.8521983623504639, "topicality": 0.8521983623504639}, {"mid": "/m/0cgh4", "description": "Building", "score": 0.8257972598075867, "topicality": 0.8257972598075867}, {"mid": "/m/03f6tq", "description": "Living room", "score": 0.8253383040428162, "topicality": 0.8253383040428162}, {"mid": "/m/0l7_8", "description": "Floor", "score": 0.822767972946167, "topicality": 0.822767972946167}, {"mid": "/m/09qqq", "description": "Wall", "score": 0.8214634656906128, "topicality": 0.8214634656906128}, {"mid": "/m/01c34b", "description": "Flooring", "score": 0.8167601227760315, "topicality": 0.8167601227760315}]}
{"photo_id": "YQsWwBD7amU8anoC14_r7g", "business_id": "kxX2SOes4o-D3ZQBkiMRfA", "labels": [{"mid": "/m/0ch399z", "description": "Drinking establishment", "score": 0.9647241234779358, "topicality": 0.9647241234779358}, {"mid": "/m/04dr76w", "description": "Bottle", "score": 0.9271307587623596, "topicality": 0.9271307587623596}, {"mid": "/m/050h26", "description": "Drinkware", "score": 0.9164530634880066, "topicality": 0.9164530634880066}, {"mid": "/m/0h8nvcj", "description": "Barware", "score": 0.8952100276947021, "topicality": 0.8952100276947021}, {"mid": "/m/0cgh4", "description": "Building", "score": 0.833665668964386, "topicality": 0.833665668964386}, {"mid": "/m/012mj", "description": "Alcoholic beverage", "score": 0.8165436387062073, "topicality": 0.8165436387062073}, {"mid": "/m/0271t", "description": "Drink", "score": 0.8124020099639893, "topicality": 0.8124020099639893}, {"mid": "/m/04bcr3", "description": "Table", "score": 0.7686349749565125, "topicality": 0.7686349749565125}, {"mid": "/m/0gjbg72", "description": "Shelf", "score": 0.7326837182044983, "topicality": 0.7326837182044983}, {"mid": "/m/04shl0", "description": "Distilled beverage", "score": 0.6895914673805237, "topicality": 0.6895914673805237}]}
{"photo_id": "95W4pZyVQgaU6ZRliv1Pgw", "business_id": "e4Vwtrqf-wpJfwesgvdgxQ", "labels": [{"mid": "/m/0c_jw", "description": "Furniture", "score": 0.9547609090805054, "topicality": 0.9547609090805054}, {"mid": "/m/0ch399z", "description": "Drinking establishment", "score": 0.9495112299919128, "topicality": 0.9495112299919128}, {"mid": "/m/04bcr3", "description": "Table", "score": 0.9293937087059021, "topicality": 0.9293937087059021}, {"mid": "/m/0h8nvcj", "description": "Barware", "score": 0.8801243305206299, "topicality": 0.8801243305206299}, {"mid": "/m/01mzpv", "description": "Chair", "score": 0.8616512417793274, "topicality": 0.8616512417793274}, {"mid": "/m/0c3cnc", "description": "Bar stool", "score": 0.8007789850234985, "topicality": 0.8007789850234985}, {"mid": "/m/0cgh4", "description": "Building", "score": 0.7569829225540161, "topicality": 0.7569829225540161}, {"mid": "/m/01nz0l", "description": "Tavern", "score": 0.6663479804992676, "topicality": 0.6663479804992676}, {"mid": "/m/03gfsp", "description": "Ceiling", "score": 0.6470767259597778, "topicality": 0.6470767259597778}, {"mid": "/m/09s1f", "description": "Business", "score": 0.6442829370498657, "topicality": 0.6442829370498657}]}"""


@pytest.fixture(scope="function")
def yelp_reviews_raw() -> str:
    return """{"review_id":"KU_O5udG6zpxOg-VcAEodg","user_id":"mh_-eMZ6K5RLWhZyISBhwA","business_id":"XQfwVwDr-v0ZS3_CbbE5Xw","stars":3.0,"useful":0,"funny":0,"cool":0,"text":"If you decide to eat here, just be aware it is going to take about 2 hours from beginning to end. We have tried it multiple times, because I want to like it! I have been to it's other locations in NJ and never had a bad experience. \\n\\nThe food is good, but it takes a very long time to come out. The waitstaff is very young, but usually pleasant. We have just had too many experiences where we spent way too long waiting. We usually opt for another diner or restaurant on the weekends, in order to be done quicker.","date":"2018-07-07 22:09:11"}
{"review_id":"BiTunyQ73aT9WBnpR9DZGw","user_id":"OyoGAe7OKpv6SyGZT5g77Q","business_id":"7ATYjTIgM3jUlt4UM3IypQ","stars":5.0,"useful":1,"funny":0,"cool":1,"text":"I've taken a lot of spin classes over the years, and nothing compares to the classes at Body Cycle. From the nice, clean space and amazing bikes, to the welcoming and motivating instructors, every class is a top notch work out.\\n\\nFor anyone who struggles to fit workouts in, the online scheduling system makes it easy to plan ahead (and there's no need to line up way in advanced like many gyms make you do).\\n\\nThere is no way I can write this review without giving Russell, the owner of Body Cycle, a shout out. Russell's passion for fitness and cycling is so evident, as is his desire for all of his clients to succeed. He is always dropping in to classes to check in/provide encouragement, and is open to ideas and recommendations from anyone. Russell always wears a smile on his face, even when he's kicking your butt in class!","date":"2012-01-03 15:28:18"}
{"review_id":"saUsX_uimxRlCVr67Z4Jig","user_id":"8g_iMtfSiwikVnbP2etR0A","business_id":"YjUWPpI6HXG530lwP-fb2A","stars":3.0,"useful":0,"funny":0,"cool":0,"text":"Family diner. Had the buffet. Eclectic assortment: a large chicken leg, fried jalapeño, tamale, two rolled grape leaves, fresh melon. All good. Lots of Mexican choices there. Also has a menu with breakfast served all day long. Friendly, attentive staff. Good place for a casual relaxed meal with no expectations. Next to the Clarion Hotel.","date":"2014-02-05 20:30:30"}
{"review_id":"AqPFMleE6RsU23_auESxiA","user_id":"_7bHUi9Uuf5__HHc_Q8guQ","business_id":"kxX2SOes4o-D3ZQBkiMRfA","stars":5.0,"useful":1,"funny":0,"cool":1,"text":"Wow!  Yummy, different,  delicious.   Our favorite is the lamb curry and korma.  With 10 different kinds of naan!!!  Don't let the outside deter you (because we almost changed our minds)...go in and try something new!   You'll be glad you did!","date":"2015-01-04 00:01:03"}
{"review_id":"Sx8TMOWLNuJBWer-0pcmoA","user_id":"bcjbaE6dDog4jkNY91ncLQ","business_id":"e4Vwtrqf-wpJfwesgvdgxQ","stars":4.0,"useful":1,"funny":0,"cool":1,"text":"Cute interior and owner (?) gave us tour of upcoming patio/rooftop area which will be great on beautiful days like today. Cheese curds were very good and very filling. Really like that sandwiches come w salad, esp after eating too many curds! Had the onion, gruyere, tomato sandwich. Wasn't too much cheese which I liked. Needed something else...pepper jelly maybe. Would like to see more menu options added such as salads w fun cheeses. Lots of beer and wine as well as limited cocktails. Next time I will try one of the draft wines.","date":"2017-01-14 20:54:15"}"""


def test_reviews_dataset(
    raw_data_root: Path,
    clean_data_root,
    image_tags_raw: str,
    yelp_reviews_raw: str,
) -> None:
    images_columns = [
        "image_id",
        "restaurant_id",
        "tag_id",
        "weight",
        "image_index",
        "tag_index",
    ]
    reviews_columns = [
        "user_id",
        "rating",
        "timestamp",
        "image_id",
        "image_index",
        "weight",
        "user_index",
    ]

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
            train_size=0.8,
            n_folds=4,
            make_if_not_exists=True,
        )

    image_tags_path.parent.mkdir(exist_ok=True, parents=True)
    with open(image_tags_path, "w+") as f:
        f.write(image_tags_raw)

    with pytest.raises(FileNotFoundError):
        YelpReviews(
            images_path=clean_data_root / "yelp_images_2.csv",
            reviews_path=clean_data_root / "yelp_reviews_2.csv",
            image_tags_path=image_tags_path,
            yelp_reviews_path=yelp_reviews_path,
            train_size=0.8,
            n_folds=4,
            make_if_not_exists=True,
        )

    yelp_reviews_path.parent.mkdir(exist_ok=True, parents=True)
    with open(yelp_reviews_path, "w+") as f:
        f.write(yelp_reviews_raw)

    yelp_reviews_1 = YelpReviews(
        images_path=clean_data_root / "yelp_images.csv",
        reviews_path=clean_data_root / "yelp_reviews.csv",
        image_tags_path=image_tags_path,
        yelp_reviews_path=yelp_reviews_path,
        train_size=0.8,
        n_folds=4,
        make_if_not_exists=True,
    )

    assert isinstance(yelp_reviews_1.images, pd.DataFrame)
    assert list(yelp_reviews_1.images.columns) == images_columns
    assert len(yelp_reviews_1.images) == 60

    assert isinstance(yelp_reviews_1.reviews_train, pd.DataFrame)
    assert list(yelp_reviews_1.reviews_train.columns) == reviews_columns
    assert yelp_reviews_1.reviews_train.weight.sum() == 4

    assert isinstance(yelp_reviews_1.reviews_test, pd.DataFrame)
    assert list(yelp_reviews_1.reviews_test.columns) == reviews_columns
    assert yelp_reviews_1.reviews_test.weight.sum() == 1

    assert len(yelp_reviews_1.reviews_folds) == 4
    for fold in yelp_reviews_1.reviews_folds:
        assert isinstance(fold.train, pd.DataFrame)
        assert list(fold.train.columns) == reviews_columns
        assert fold.train.weight.sum() == 3

        assert isinstance(fold.test, pd.DataFrame)
        assert list(fold.test.columns) == reviews_columns
        assert fold.test.weight.sum() == 1

    with pytest.raises(FileNotFoundError):
        YelpReviews(
            images_path=clean_data_root / "yelp_images_3.csv",
            reviews_path=clean_data_root / "yelp_reviews_3.csv",
            train_size=0.8,
            n_folds=4,
        )

    with pytest.raises(FileNotFoundError):
        YelpReviews(
            images_path=clean_data_root / "yelp_images.csv",
            reviews_path=clean_data_root / "yelp_reviews_4.csv",
            train_size=0.8,
            n_folds=4,
        )

    yelp_reviews_2 = YelpReviews(
        images_path=clean_data_root / "yelp_images.csv",
        reviews_path=clean_data_root / "yelp_reviews.csv",
        train_size=0.8,
        n_folds=4,
    )

    assert yelp_reviews_2.images.equals(yelp_reviews_1.images)
    assert yelp_reviews_2.reviews_train.equals(yelp_reviews_1.reviews_train)
    assert yelp_reviews_2.reviews_test.equals(yelp_reviews_1.reviews_test)
    assert len(yelp_reviews_2.reviews_folds) == 4
    for fold_2, fold_1 in zip(
        yelp_reviews_2.reviews_folds, yelp_reviews_1.reviews_folds
    ):
        assert fold_2.train.equals(fold_1.train)
        assert fold_2.test.equals(fold_1.test)


def test_train_test_splidd() -> None:
    pass


def test_k_folds_splidd() -> None:
    pass
