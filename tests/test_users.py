import pytest
from src.db import User
from src.users import UserService


def test_sign_up(user_service: UserService) -> None:
    with pytest.raises(ValueError):
        user_service.sign_up("a@gmail.com", "a")

    user_service.sign_up("b@gmail.com", "b")


def test_sign_in(user_service: UserService) -> None:
    with pytest.raises(ValueError):
        user_service.sign_in("a@gmail.com", "b")

    with pytest.raises(ValueError):
        user_service.sign_in("c@gmail.com", "c")

    user = user_service.sign_in("a@gmail.com", "a")
    assert isinstance(user, User)
    assert user.email == "a@gmail.com"
