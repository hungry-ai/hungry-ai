import pytest

from src.frontend import Frontend


def test_story_mention(frontend: Frontend) -> None:
    with pytest.raises(ValueError):
        frontend.story_mention("cody", "food.jpg", 6)

    with pytest.raises(ValueError):
        frontend.story_mention("cody", "food.jpg", 0)

    assert frontend.story_mention("cody", "food.jpg", 5) == "review added"


def test_search(frontend: Frontend) -> None:
    assert frontend.search("asdf", "", "") == "queries not supported yet"
    assert frontend.search("", "asdf", "") == "locations not supported yet"
    assert (
        frontend.search("", "", "cody")
        == "\"user with instagram_username='cody' does not exist\""
    )

    frontend.story_mention("cody", "food.jpg", 5)

    print(frontend.search("", "", "cody"))

    assert (
        frontend.search("", "", "cody")
        == """\
Recommended images:
\tfood.jpg
My reviews:
\t5 - food.jpg
My stats:
No stats available."""
    )
