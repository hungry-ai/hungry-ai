from db import Topic, TopicDB


class TopicService:
    def __init__(self, topic_db: TopicDB) -> None:
        self.topic_db = topic_db

    def get_topics(self) -> list[Topic]:
        return self.topic_db.select()
