import pytest

from src.images import Image, ImageService


def test_image_service(image_service: ImageService, url: str) -> None:
    banana = image_service.add_image(url)

    assert isinstance(banana, Image)
    assert banana.url == url

    assert image_service.get_image(banana.id) == banana

    with pytest.raises(KeyError):
        image_service.get_image("bad_image_id")
