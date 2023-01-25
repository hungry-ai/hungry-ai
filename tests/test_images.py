from src import Image, ImageDB, ImageService
import pytest


@pytest.fixture(scope="module")
def image_db() -> ImageDB:
    return ImageDB()


@pytest.fixture(scope="module")
def image_service(image_db: ImageDB) -> ImageService:
    return ImageService(image_db)


def test_add(image_service: ImageService) -> None:
    with pytest.raises(ValueError):
        image_service.get_image("bad_id")

    url = "https://i5.walmartimages.com/asr/5939a6fa-a0d6-431c-88c6-b4f21608e4be.f7cd0cc487761d74c69b7731493c1581.jpeg"
    image_id = image_service.add_image(url)
    image = image_service.get_image(image_id)
    assert isinstance(image, Image)
