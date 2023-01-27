import datetime
from dataclasses import dataclass, fields
from uuid import uuid4
from hashlib import sha256
from pathlib import Path

from typing import Any, Generic, TypeVar

import pandas as pd


@dataclass
class DBSchema:
    pass


RowT = TypeVar("RowT", bound=DBSchema)


class DB(Generic[RowT]):
    def __init__(self, path: Path) -> None:
        self.path = path
        print("Using", self.__class__.__name__, "at path:", self.path)

    @property
    def cls(self) -> type:  # TODO: get rid of this
        raise NotImplementedError

    @property
    def columns(self) -> list[str]:
        return [field.name for field in fields(self.cls)]

    @property
    def df(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=self.columns)
        return pd.read_csv(self.path)

    def select(self, **where: Any) -> list[RowT]:
        return [
            self.cls(*[row[k] for k in self.columns])
            for _, row in self.df.iterrows()
            if all([row[k] == v for k, v in where.items()])
        ]

    def insert(self, row: RowT) -> None:
        df = self.df.append(
            {column: getattr(row, column) for column in self.columns},
            ignore_index=True,
        )
        df.to_csv(self.path, index=False)


@dataclass
class User(DBSchema):
    user_id: str
    email: str
    password_hash: str


class UserDB(DB[User]):
    @property
    def cls(self) -> type:
        return User


@dataclass
class Image(DBSchema):
    image_id: str
    url: str


class ImageDB(DB[Image]):
    @property
    def cls(self) -> type:
        return Image


@dataclass
class Review(DBSchema):
    review_id: str
    user_id: str
    image_id: str
    rating: int
    timestamp: datetime.datetime


class ReviewDB(DB[Review]):
    @property
    def cls(self) -> type:
        return Review


@dataclass
class Topic:
    topic_id: str
    words: list[str]

    @property
    def name(self) -> str:
        return " ".join(self.words)


class TopicDB(DB[Topic]):
    @property
    def cls(self) -> type:
        return Topic


@dataclass
class Graph(DBSchema):
    pass


class GraphDB(DB[Graph]):
    @property
    def cls(self) -> type:
        return Graph
