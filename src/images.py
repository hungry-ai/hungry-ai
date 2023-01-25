from db import Image, ImageDB
from topics import TopicService
from graph import GraphService

from uuid import uuid4


def pr_match(url: str, topic: str) -> float:  # TODO
    # returns the probability that an image matches a certain topic
    return 1.0


class ImageService:
    def __init__(
        self,
        image_db: ImageDB,
        topic_service: TopicService,
        graph_service: GraphService,
    ) -> None:
        self.image_db = image_db
        self.topic_service = topic_service
        self.graph_service = graph_service

    def add_image(self, url: str) -> str:
        image_id = uuid4()
        image = Image(image_id, url)
        self.image_db.insert(image)

        topics = self.topic_service.get_topics()
        for topic in topics:
            p = pr_match(image.url, topic.name)
            self.graph_service.add_image_edge(image.image_id, topic.topic_id, p)

        return image_id

    def get_image(self, image_id: str) -> Image:
        images = self.image_db.select(image_id=image_id)

        if len(images) == 0:
            raise ValueError("no image with this image id")

        assert len(images) == 1
        image = images[0]

        return image
