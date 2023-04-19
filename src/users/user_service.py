import warnings
from uuid import uuid4

from ..db import User, UserDB
from ..graph import GraphService


class UserService:
    def __init__(self, user_db: UserDB, graph_service: GraphService) -> None:
        self.user_db = user_db
        self.graph_service = graph_service

    def get_user_id(self, instagram_username: str, raise_if_empty: bool = True) -> str:
        results = self.user_db.select(instagram_username=instagram_username)

        if len(results) == 0:
            if raise_if_empty:
                raise KeyError(f"user with {instagram_username=} does not exist")

            user_id = str(uuid4())
            user = User(user_id, instagram_username)
            self.user_db.insert(user)
            self.graph_service.add_user(user_id)
            return user_id

        if len(results) > 1:
            warnings.warn("multiple users with that instagram_username")

        user_id = results[0].user_id
        return user_id
