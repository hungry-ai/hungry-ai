import datetime
from dataclasses import dataclass, fields
from hashlib import sha256
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import uuid4

import pandas as pd


@dataclass
class DBSchema:
    pass


RowT = TypeVar("RowT", bound=DBSchema)


def to_pandas_dtype(py_type: type) -> str:
    str_py_type = py_type.__name__
    if str_py_type in ("str", "int", "float"):
        return str_py_type
    elif str_py_type == "datetime":
        return "datetime64[ns]"
    raise NotImplementedError


class DB(Generic[RowT]):
    def __init__(self, path: Path) -> None:
        self.columns = [(field.name, field.type) for field in fields(self.cls)]
        self.path = path
        print("Using", self.__class__.__name__, "at path:", self.path)

    @property
    def cls(self) -> type:  # TODO: get rid of this
        raise NotImplementedError

    @property
    def df(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(
                {
                    colname: pd.Series(dtype=to_pandas_dtype(coltype))
                    for colname, coltype in self.columns
                }
            )
        return pd.read_csv(self.path)

    def select(self, **where: Any) -> list[RowT]:
        return [
            self.cls(*[row[k] for k, _ in self.columns])
            for _, row in self.df.iterrows()
            if all([row[k] == v for k, v in where.items()])
        ]

    def insert(self, row: RowT) -> None:
        df = self.df.append(
            {colname: getattr(row, colname) for colname, _ in self.columns},
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
class Topic(DBSchema):
    topic_id: str
    name: str


class TopicDB(DB[Topic]):
    @property
    def cls(self) -> type:
        return Topic


@dataclass
class Edge(DBSchema):
    from_id: str
    from_type: int
    to_id: str
    to_type: int
    weight: float
    timestamp: datetime.datetime


class EdgeDB(DB[Edge]):
    @property
    def cls(self) -> type:
        return Edge


@dataclass
class Recommendation(DBSchema):
    pass


class RecommendationDB(DB[Recommendation]):
    @property
    def cls(self) -> type:
        return Recommendation
