from uuid import uuid4
from ..db import Image, ImageDB
from ..graph import GraphService
from ..tags import TagService
from .pr_match import PrMatch


class ImageService:
    def __init__(
        self,
        image_db: ImageDB,
        tag_service: TagService,
        graph_service: GraphService,
    ) -> None:
        self.image_db = image_db
        self.tag_service = tag_service
        self.graph_service = graph_service
        self.pr_match = PrMatch()

    def add_image(self, url: str) -> str:
        image_id = str(uuid4())
        image = Image(image_id, url)
        self.image_db.insert(image)
        self.graph_service.add_image(image_id)

        tags = self.tag_service.get_tags()
        probabilities = self.pr_match(image.url, [tag.name for tag in tags])
        for p, tag in zip(probabilities, tags):
            self.graph_service.add_image_edge(image.image_id, tag.tag_id, p)

        return image_id

    def get_image(self, image_id: str) -> Image:
        images = self.image_db.select(image_id=image_id)

        if len(images) == 0:
            raise ValueError("no image with this image id")

        assert len(images) == 1
        image = images[0]

        return image
