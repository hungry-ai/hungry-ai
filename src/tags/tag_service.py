from ..db import Tag, TagDB


class TagService:
    def __init__(self, tag_db: TagDB) -> None:
        self.tag_db = tag_db

    def get_tags(self) -> list[Tag]:
        return self.tag_db.select()
