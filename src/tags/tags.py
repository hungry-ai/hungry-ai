from dataclasses import dataclass


@dataclass(frozen=True)
class Tag:
    id: str
    name: str
