import pytest

from src.users import UserService


def test_get_user_id(user_service: UserService) -> None:
    with pytest.raises(KeyError):
        user_service.get_user_id("cody")

    user_id = user_service.get_user_id("cody", raise_if_empty=False)

    assert user_service.get_user_id("cody") == user_id
