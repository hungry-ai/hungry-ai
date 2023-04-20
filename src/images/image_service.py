from uuid import uuid4

from .images import Image


class ImageService:
    def __init__(self) -> None:
        self.images: dict[str, Image] = {}  # image_id -> image

    def add_image(self, url: str) -> Image:
        image_id = str(uuid4())
        image = Image(image_id, url)
        self.images[image_id] = image

        return image

    def get_image(self, image_id: str) -> Image:
        return self.images[image_id]
