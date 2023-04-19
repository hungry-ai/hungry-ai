import pytest

from src.db import Image
from src.images import ImageService


def test_add_image(image_service: ImageService, url: str) -> None:
    image_service.add_image(url)


def test_get_image(image_service: ImageService, url: str) -> None:
    image_id = image_service.add_image(url)
    image = image_service.get_image(image_id)

    assert isinstance(image, Image)
    assert image.image_id == image_id

    with pytest.raises(ValueError):
        image_service.get_image("bad_image_id")
