import datetime
import pytest

from src.topics_graph.construct_graph_from_embedding import (
    construct_graph_from_embedding,
)
from src.topics_graph.graph_basic import GraphBasic
from src.word_embedding.word_embedding_basic import WordEmbeddingBasic


def test_construct_graph_from_embedding():
    word_embedding_1 = WordEmbeddingBasic()
    word_embedding_1.add_word_vector(("A", [1.0, 0.0, 0.0, 0.0, 0.0]))
    word_embedding_1.add_word_vector(("B", [0.0, 1.0, 0.0, 0.0, 0.0]))
    word_embedding_1.add_word_vector(("C", [0.0, 0.0, 2.0, 0.0, 0.0]))
    word_embedding_1.add_word_vector(("Hello", [0.0, 0.0, 0.0, 1.3, 0.0]))
    word_embedding_1.add_word_vector(("Bonjour", [0.0, 0.0, 0.0, 0.0, 1.0]))
    word_embedding_1.add_word_vector(("Nihao", [1.2, 0.0, 0.0, 0.0, 0.0]))
    word_embedding_1.add_word_vector(("Konichiwa", [1.3, 0.0, 0.0, 0.0, 0.0]))
    graph_1 = GraphBasic()
    construct_graph_from_embedding(word_embedding_1, graph_1)
    assert 7 == graph_1.number_of_vertices()
    assert 7 == graph_1.number_of_edges()

    edge_1 = graph_1.get_edge(0, 5)
    edge_2 = graph_1.get_edge(0, 2)
    edge_3 = graph_1.get_edge(0, 1)
    edge_4 = graph_1.get_edge(0, 3)
    edge_5 = graph_1.get_edge(0, 4)
    edge_6 = graph_1.get_edge(0, 6)

    assert None != edge_1
    assert None == edge_2
    assert None == edge_3
    assert None == edge_4
    assert None == edge_5
    assert None == edge_6

    edge_1 = graph_1.get_edge(5, 6)
    edge_2 = graph_1.get_edge(5, 0)
    edge_3 = graph_1.get_edge(5, 1)
    edge_4 = graph_1.get_edge(5, 2)
    edge_5 = graph_1.get_edge(5, 3)
    edge_6 = graph_1.get_edge(5, 4)

    assert None != edge_1
    assert None == edge_2
    assert None == edge_3
    assert None == edge_4
    assert None == edge_5
    assert None == edge_6
