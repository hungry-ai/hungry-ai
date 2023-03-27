import sys

sys.path.append('../word_embedding')

from word_embedding_base import WordEmbeddingBase
from graph_base import GraphBase

def construct_graph_from_embedding(word_embedding: WordEmbeddingBase, graph: GraphBase):
    for i in range(word_embedding.number_of_words()):
        graph.add_vertex(word_embedding.get_ith_word_vector(i))
    
    for i in range(word_embedding.number_of_words()):
        min_distance = None
        closest_neighbor = None
        for j in range(word_embedding.number_of_words()):
            if i == j: continue
            current_distance = word_embedding.get_distance(i, j)
            if min_distance == None or current_distance < min_distance:
                min_distance = current_distance
                closest_neighbor = j
        # Figure out how to determine weight of edge.
        graph.add_edge((i, closest_neighbor, 1.0))


    
