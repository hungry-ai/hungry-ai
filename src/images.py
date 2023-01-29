from uuid import uuid4
from .db import Image, ImageDB
from .graph import GraphService
from .topics import TopicService
from .pr_match import PrMatch


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
        self.pr_match = PrMatch()

    def add_image(self, url: str) -> str:
        image_id = str(uuid4())
        image = Image(image_id, url)
        self.image_db.insert(image)

        topics = self.topic_service.get_topics()
        probabilities = self.pr_match(image.url, [topic.name for topic in topics])
        for p, topic in zip(probabilities, topics):
            self.graph_service.add_image_edge(image.image_id, topic.topic_id, p)

        return image_id

    def get_image(self, image_id: str) -> Image:
        images = self.image_db.select(image_id=image_id)

        if len(images) == 0:
            raise ValueError("no image with this image id")

        assert len(images) == 1
        image = images[0]

        return image
