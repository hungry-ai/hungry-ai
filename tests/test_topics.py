from pathlib import Path

from src.db import Topic
from src.topics import TopicService


def test_get_topics(topic_service: TopicService) -> None:
    topics = topic_service.get_topics()

    assert isinstance(topics, list)
    assert len(topics) == 2

    assert isinstance(topics[0], Topic)
    assert topics[0].topic_id == "t1"
    assert topics[0].name == "hello world"

    assert isinstance(topics[1], Topic)
    assert topics[1].topic_id == "t2"
    assert topics[1].name == "hi"
