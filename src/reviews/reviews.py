import datetime
from dataclasses import dataclass

from ..images import Image
from ..users import User


@dataclass(frozen=True)
class Review:
    id: str
    user: User
    image: Image
    rating: int
    timestamp: datetime.datetime
