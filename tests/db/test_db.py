import datetime
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.db import DB, DBSchema


@dataclass
class Test(DBSchema):
    a: int
    b: str
    c: float
    d: datetime.datetime


class TestDB(DB[Test]):
    @property
    def cls(self) -> type:
        return Test


@pytest.fixture(scope="function")
def test_db(root: Path) -> TestDB:
    return TestDB(root / "test.csv")


def test_insert(test_db: TestDB) -> None:
    row_1 = Test(1, "hi", 0.0, datetime.datetime.now())
    row_2 = Test(2, "hello", 1.0, datetime.datetime.now())

    test_db.insert(row_1)
    test_db.insert(row_2)


def test_select(test_db: TestDB) -> None:
    dt_1 = datetime.datetime.now()
    dt_2 = datetime.datetime.now()

    row_1 = Test(1, "hi", 0.0, dt_1)
    row_2 = Test(2, "hi", 1.0, dt_2)
    row_3 = Test(3, "sup", 2.0, dt_2)

    assert len(test_db.select()) == 0
    test_db.insert(row_1)
    assert len(test_db.select()) == 1
    test_db.insert(row_2)
    assert len(test_db.select()) == 2
    test_db.insert(row_3)

    assert test_db.select() == [row_1, row_2, row_3]
    assert test_db.select(a=1) == [row_1]
    assert test_db.select(b="hi") == [row_1, row_2]
    assert test_db.select(d=str(dt_2)) == [row_2, row_3]

    assert test_db.select(b="hi", d=str(dt_1)) == [row_1]
    assert test_db.select(b="yo") == []
    assert test_db.select(b="sup", d=dt_1) == []


def test_df(test_db: TestDB) -> None:
    row_1 = Test(1, "hi", 0.0, datetime.datetime.now())
    row_2 = Test(2, "hi", 1.0, datetime.datetime.now())
    row_3 = Test(3, "sup", 2.0, datetime.datetime.now())

    test_db.insert(row_1)
    test_db.insert(row_2)
    test_db.insert(row_3)

    df = test_db.df

    assert list(df.columns) == ["a", "b", "c", "d"]
    assert list(df.dtypes) == [
        np.dtype("int64"),
        np.dtype("O"),
        np.dtype("float64"),
        np.dtype("O"),
    ]
    assert len(df) == 3
