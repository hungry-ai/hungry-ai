from src.images import pr_match


def test_pr_match() -> None:
    url = "https://upload.wikimedia.org/wikipedia/commons/8/89/Tomato_je.jpg"
    tag = "tomato"

    # TODO: comment this out because pr_match will cost money to run
    assert 0.0 <= pr_match(url, tag) <= 1.0
