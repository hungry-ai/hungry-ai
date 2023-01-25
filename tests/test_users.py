from src import UserDB, UserService
import pytest


@pytest.fixture(scope="module")
def user_db() -> UserDB:
    return UserDB()


@pytest.fixture(scope="module")
def user_service(user_db: UserDB) -> UserService:
    return UserService(user_db)


def test_registration(user_service: UserService) -> None:
    user_service.sign_up("cody@gmail.com", "cody")

    with pytest.raises(ValueError):
        user_service.sign_up("cody@gmail.com", "nody")

    with pytest.raises(ValueError):
        user_service.sign_in("cody@gmail.com", "nody")

    user_service.sign_in("cody@gmail.com", "cody")
