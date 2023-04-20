from dataclasses import dataclass


@dataclass(frozen=True)
class User:
    id: str
    instagram_username: str
