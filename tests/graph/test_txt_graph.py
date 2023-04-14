from src.graph import TXTGraph
from pathlib import Path


def test_txt_graph(root: Path) -> None:
    filename = root / "test_1.txt"
    graph_1 = TXTGraph(filename)


def test_read_graph() -> None:
    assert False


def test_write_graph() -> None:
    assert False
