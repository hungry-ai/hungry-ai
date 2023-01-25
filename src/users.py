from db import User, UserDB

from uuid import uuid4
from hashlib import sha256


class UserService:
    def __init__(self, user_db: UserDB) -> None:
        self.user_db = user_db

    def hash(self, x: str) -> str:
        return sha256(x.encode("utf-8")).hexdigest()

    def sign_up(self, email: str, password: str) -> None:
        users = self.user_db.select(email=email)

        if len(users) > 0:
            raise ValueError("user with this email already exists")

        user_id = uuid4()
        password_hash = self.hash(password)
        user = User(user_id, email, password_hash)
        self.user_db.insert(user)

    def sign_in(self, email: str, password: str) -> User:
        users = self.user_db.select(email=email)

        if len(users) == 0:
            raise ValueError("no user with this email")

        assert len(users) == 1
        user = users[0]

        if user.password_hash != self.hash(password):
            raise ValueError("incorrect password")

        return user
