from dataclasses import dataclass
from typing import Tuple

from ..tags import Tag


@dataclass(frozen=True)
class Image:
    id: str
    url: str


@dataclass(frozen=True)
class TaggedImage(Image):
    tags: list[Tuple[Tag, float]]
