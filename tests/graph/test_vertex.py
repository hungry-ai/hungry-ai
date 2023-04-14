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
    vertex = Vertex("u", VertexType.USER)

    assert isinstance(vertex.id, str)
    assert isinstance(vertex.type, VertexType)
    assert vertex.name == "user_u"
