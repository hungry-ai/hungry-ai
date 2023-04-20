import pytest

from src.graph import Vertex, VertexType


def test_vertex_type() -> None:
    assert VertexType.USER.value == 0
    assert VertexType.USER == VertexType(0)
    with pytest.raises(ValueError):
        VertexType(-1)
    with pytest.raises(ValueError):
        VertexType(3)
    assert len({VertexType.USER, VertexType.IMAGE, VertexType.TAG}) == 3


def test_vertex() -> None:
    vertex = Vertex("my_user", VertexType.USER)

    assert vertex.id == "my_user"
    assert vertex.type == VertexType.USER
    assert isinstance(vertex.type, VertexType)
    assert vertex.name == "user_my_user"
    assert vertex.__dataclass_params__.frozen  # type: ignore

    vertex = Vertex("my_image", VertexType.IMAGE)

    assert vertex.id == "my_image"
    assert vertex.type == VertexType.IMAGE
    assert isinstance(vertex.type, VertexType)
    assert vertex.name == "image_my_image"
    assert vertex.__dataclass_params__.frozen  # type: ignore

    vertex = Vertex("my_tag", VertexType.TAG)

    assert vertex.id == "my_tag"
    assert vertex.type == VertexType.TAG
    assert isinstance(vertex.type, VertexType)
    assert vertex.name == "tag_my_tag"
    assert vertex.__dataclass_params__.frozen  # type: ignore
