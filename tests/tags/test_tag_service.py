from src.db import Tag
from src.tags import TagService


def test_get_tags(tag_service: TagService) -> None:
    tags = tag_service.get_tags()

    assert isinstance(tags, list)
    assert len(tags) == 2

    assert isinstance(tags[0], Tag)
    assert tags[0].tag_id == "t1"
    assert tags[0].name == "hello world"

    assert isinstance(tags[1], Tag)
    assert tags[1].tag_id == "t2"
    assert tags[1].name == "hi"
