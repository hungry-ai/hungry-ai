from src.tags import Tag


def test_tags(japanese: Tag, soup: Tag, ramen: Tag) -> None:
    assert japanese.id == "_japanese"
    assert japanese.name == "japanese"
    assert japanese.__dataclass_params__.frozen  # type: ignore

    assert soup.id == "_soup"
    assert soup.name == "soup"
    assert soup.__dataclass_params__.frozen  # type: ignore

    assert ramen.id == "_ramen"
    assert ramen.name == "ramen"
    assert ramen.__dataclass_params__.frozen  # type: ignore
