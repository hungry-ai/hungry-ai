from topics_graph.graph_base import GraphBase
from word_embedding.word_embedding_base import WordEmbeddingBase


def construct_graph_from_embedding(word_embedding: WordEmbeddingBase, graph: GraphBase):
    for i in range(word_embedding.number_of_words()):
        word, word_vector = word_embedding.get_ith_word_vector(i)
        graph.add_vertex((2, word, word_vector))

    for i in range(word_embedding.number_of_words()):
        min_distance = None
        closest_neighbor = None
        for j in range(word_embedding.number_of_words()):
            if i == j:
                continue
            current_distance = word_embedding.get_distance(i, j)
            if min_distance == None or current_distance < min_distance:
                min_distance = current_distance
                closest_neighbor = j
        # Figure out how to determine weight of edge.
        current_word = word_embedding.get_ith_word_vector(i)[0]
        closest_word = word_embedding.get_ith_word_vector(closest_neighbor)[0]
        if current_word == closest_word: 
            print("There's a self-loop! " + str(closest_neighbor) + " " + str(i) + " " + current_word)
        graph.add_edge((current_word, closest_word, 1.0))
