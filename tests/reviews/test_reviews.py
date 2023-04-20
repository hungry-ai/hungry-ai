import datetime

from src.images import Image
from src.reviews import Review
from src.users import User


def test_reviews(
    cody_tonkotsu: Review,
    alex_tonkotsu: Review,
    alex_chicken_noodle: Review,
    younes_sushi_1: Review,
    younes_sushi_2: Review,
    cody: User,
    alex: User,
    younes: User,
    tonkotsu: Image,
    chicken_noodle: Image,
    sushi: Image,
) -> None:
    assert cody_tonkotsu.id == "_cody_tonkotsu"
    assert cody_tonkotsu.user == cody
    assert cody_tonkotsu.image == tonkotsu
    assert cody_tonkotsu.rating == 5.0
    assert cody_tonkotsu.timestamp == datetime.datetime(2023, 1, 1)
    assert cody_tonkotsu.__dataclass_params__.frozen  # type: ignore

    assert alex_tonkotsu.id == "_alex_tonkotsu"
    assert alex_tonkotsu.user == alex
    assert alex_tonkotsu.image == tonkotsu
    assert alex_tonkotsu.rating == 5.0
    assert alex_tonkotsu.timestamp == datetime.datetime(2023, 1, 1)
    assert alex_tonkotsu.__dataclass_params__.frozen  # type: ignore

    assert alex_chicken_noodle.id == "_alex_chicken_noodle"
    assert alex_chicken_noodle.user == alex
    assert alex_chicken_noodle.image == chicken_noodle
    assert alex_chicken_noodle.rating == 1.0
    assert alex_chicken_noodle.timestamp == datetime.datetime(2023, 1, 2)
    assert alex_chicken_noodle.__dataclass_params__.frozen  # type: ignore

    assert younes_sushi_1.id == "_younes_sushi_1"
    assert younes_sushi_1.user == younes
    assert younes_sushi_1.image == sushi
    assert younes_sushi_1.rating == 5.0
    assert younes_sushi_1.timestamp == datetime.datetime(2023, 1, 1)
    assert younes_sushi_1.__dataclass_params__.frozen  # type: ignore

    assert younes_sushi_2.id == "_younes_sushi_2"
    assert younes_sushi_2.user == younes
    assert younes_sushi_2.image == sushi
    assert younes_sushi_2.rating == 4.0
    assert younes_sushi_2.timestamp == datetime.datetime(2023, 1, 2)
    assert younes_sushi_2.__dataclass_params__.frozen  # type: ignore
