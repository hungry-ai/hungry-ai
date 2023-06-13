from dataclasses import dataclass
from ..tags import Tag


@dataclass(frozen=True)
class Image:
    id: str
    url: str


@dataclass(frozen=True)
class TaggedImage(Image):
    tags: dict[Tag, float]
