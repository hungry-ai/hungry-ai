from src.images import Image


def test_images(tonkotsu: Image, chicken_noodle: Image, sushi: Image) -> None:
    assert tonkotsu.id == "_tonkotsu"
    assert (
        tonkotsu.url
        == "https://a0.cdn.japantravel.com/photo/31400-157114/1440x960!/tokyo-tonchin-ramen-157114.jpg"
    )
    assert tonkotsu.__dataclass_params__.frozen  # type: ignore

    assert chicken_noodle.id == "_chicken_noodle"
    assert (
        chicken_noodle.url
        == "https://www.averiecooks.com/wp-content/uploads/2015/09/chickennoodlesoup-14.jpg"
    )
    assert chicken_noodle.__dataclass_params__.frozen  # type: ignore

    assert sushi.id == "_sushi"
    assert (
        sushi.url
        == "https://pyxis.nymag.com/v1/imgs/9e3/a79/80a32b4009084c8c3cdb7fff4f1bc7d407-sushi-nakazawa-03.rsocial.w1200.jpg"
    )
    assert sushi.__dataclass_params__.frozen  # type: ignore
