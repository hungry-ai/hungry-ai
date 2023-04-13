from hashlib import sha256
from uuid import uuid4

from ..db import User, UserDB
from ..graph import GraphService


def hash(x: str) -> str:
    return sha256(x.encode("utf-8")).hexdigest()


class UserService:
    def __init__(self, user_db: UserDB, graph_service: GraphService) -> None:
        self.user_db = user_db
        self.graph_service = graph_service

    def sign_up(self, email: str, password: str) -> None:
        users = self.user_db.select(email=email)

        if len(users) > 0:
            raise ValueError("user with this email already exists")

        user_id = str(uuid4())
        password_hash = hash(password)
        user = User(user_id, email, password_hash)
        self.user_db.insert(user)
        self.graph_service.add_user(user_id)

    def sign_in(self, email: str, password: str) -> User:
        users = self.user_db.select(email=email)

        if len(users) == 0:
            raise ValueError("no user with this email")

        assert len(users) == 1
        user = users[0]

        if user.password_hash != hash(password):
            raise ValueError("incorrect password")

        return user
