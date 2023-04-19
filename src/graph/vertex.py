from dataclasses import dataclass
from enum import Enum


class VertexType(Enum):
    USER = 0
    IMAGE = 1
    TAG = 2


@dataclass(frozen=True)
class Vertex:
    id: str
    type: VertexType

    @property
    def name(self) -> str:
        return f"{self.type.name.lower()}_{self.id}"
