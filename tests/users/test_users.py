from src.users import User


def test_users(cody: User, alex: User, younes: User) -> None:
    assert cody.id == "_cody"
    assert cody.instagram_username == "cody"
    assert cody.__dataclass_params__.frozen  # type: ignore

    assert alex.id == "_alex"
    assert alex.instagram_username == "alex"
    assert alex.__dataclass_params__.frozen  # type: ignore

    assert younes.id == "_younes"
    assert younes.instagram_username == "younes"
    assert younes.__dataclass_params__.frozen  # type: ignore
