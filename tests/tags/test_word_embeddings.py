import numpy as np
import pytest

from src.tags import WordEmbedding


def test_word_embedding():
    word_embedding_1 = WordEmbedding(dimension=5)
    assert 5 == word_embedding_1.dimension
    assert 0 == len(word_embedding_1)

    word_embedding_1["A"] = np.array([1, 0, 0, 0, 0])
    word_embedding_1["B"] = np.array([0, 1, 0, 0, 0])
    word_embedding_1["C"] = np.array([0, 0, 1, 0, 0])
    word_embedding_1["Hello"] = np.array([0, 0, 0, 1, 0])
    word_embedding_1["Bonjour"] = np.array([0, 0, 0, 0, 1])
    word_embedding_1["Nihao"] = np.array([1, 0, 0, 0, 1])
    word_embedding_1["Konichiwa"] = np.array([1, 0, 1, 0, 0])
    assert 7 == len(word_embedding_1)

    word_vector = word_embedding_1["Hello"]
    np.testing.assert_array_equal(word_vector, np.array([0, 0, 0, 1, 0]))
    with pytest.raises(KeyError):
        word_embedding_1["Wassup"]


def test_pytorch_word_embedding() -> None:
    assert False
