import numpy as np

from src.graph import LocalGraph
from src.tags.gen_tags import WordEmbedding, generate_tags_graph


def test_generate_tags_graph():
    word_embedding_1 = WordEmbedding(dimension = 5)
    word_embedding_1["A"] =  np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    word_embedding_1["B"] =  np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    word_embedding_1["C"] =  np.array([0.0, 0.0, 2.0, 0.0, 0.0])
    word_embedding_1["Hello"] =  np.array([0.0, 0.0, 0.0, 1.3, 0.0])
    word_embedding_1["Bonjour"] =  np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    word_embedding_1["Nihao"] =  np.array([1.2, 0.0, 0.0, 0.0, 0.0])
    word_embedding_1["Konichiwa"] =  np.array([1.3, 0.0, 0.0, 0.0, 0.0])
    graph_1 = LocalGraph()
    word_map = generate_tags_graph(word_embedding_1, graph_1)
    assert 7 == len(graph_1.vertices)

    assert word_map["Nihao"] in graph_1.out_neighbors(word_map["A"])
    assert word_map["B"] not in graph_1.out_neighbors(word_map["A"])
    assert word_map["C"] not in graph_1.out_neighbors(word_map["A"])
    assert word_map["Hello"] not in graph_1.out_neighbors(word_map["A"])
    assert word_map["Bonjour"] not in graph_1.out_neighbors(word_map["A"])
    assert word_map["Konichiwa"] not in graph_1.out_neighbors(word_map["A"])

    assert word_map["A"] not in graph_1.out_neighbors(word_map["Nihao"])
    assert word_map["B"] not in graph_1.out_neighbors(word_map["Nihao"])
    assert word_map["C"] not in graph_1.out_neighbors(word_map["Nihao"])
    assert word_map["Hello"] not in graph_1.out_neighbors(word_map["Nihao"])
    assert word_map["Bonjour"] not in graph_1.out_neighbors(word_map["Nihao"])
    assert word_map["Konichiwa"] in graph_1.out_neighbors(word_map["Nihao"])

