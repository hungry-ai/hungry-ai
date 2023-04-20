from uuid import uuid4

from .users import User


class UserService:
    def __init__(self) -> None:
        self.users: dict[str, User] = {}  # instagram_username -> user

    def add_user(self, instagram_username: str) -> User:
        user_id = str(uuid4())
        user = User(user_id, instagram_username)
        self.users[instagram_username] = user
        return user

    def get_user(self, instagram_username: str) -> User:
        return self.users[instagram_username]
