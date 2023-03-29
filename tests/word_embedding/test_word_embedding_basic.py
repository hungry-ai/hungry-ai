import datetime

import pytest

from src.word_embedding.word_embedding_basic import WordEmbeddingBasic


def test_graph_basic():
    word_embedding_1 = WordEmbeddingBasic()
    assert 5 == word_embedding_1.get_dimension()
    assert 0 == word_embedding_1.number_of_words()

    word_embedding_1.add_word_vector(("A", [1, 0, 0, 0, 0]))
    word_embedding_1.add_word_vector(("B", [0, 1, 0, 0, 0]))
    word_embedding_1.add_word_vector(("C", [0, 0, 1, 0, 0]))
    word_embedding_1.add_word_vector(("Hello", [0, 0, 0, 1, 0]))
    word_embedding_1.add_word_vector(("Bonjour", [0, 0, 0, 0, 1]))
    word_embedding_1.add_word_vector(("Nihao", [1, 0, 0, 0, 1]))
    word_embedding_1.add_word_vector(("Konichiwa", [1, 0, 1, 0, 0]))
    assert 7 == word_embedding_1.number_of_words()

    word_vector = word_embedding_1.get_word_vector("Hello")
    assert "Hello" == word_vector[0]
    assert [0, 0, 0, 1, 0] == word_vector[1]
    word_vector = word_embedding_1.get_word_vector("Wassup")
    assert None == word_vector
    word_vector = word_embedding_1.get_ith_word_vector(8)
    assert None == word_vector
    word_vector = word_embedding_1.get_ith_word_vector(6)
    assert "Konichiwa" == word_vector[0]
    assert [1, 0, 1, 0, 0] == word_vector[1]
