import pytest

from src.users import User, UserService


def test_user_service(user_service: UserService) -> None:
    with pytest.raises(KeyError):
        user_service.get_user("cody")

    cody = user_service.add_user("cody")
    assert isinstance(cody, User)

    assert user_service.get_user("cody") == cody
