from pyvis.network import Network
import networkx as nx
import torchtext
from word_embedding.word_embedding_basic import WordEmbeddingBasic
from word_embedding.word_embedding_base import WordEmbeddingBase
import topics_utils
import topics_graph.graph_txt as graph_txt
from topics_graph.graph_base import GraphBase

def graph_visualization(obj, file_name='graph_visualization', scaled=True):
    '''
    Converts various objects into pyvis network object and outputs visualization as html.
    obj : file name, OR word list, OR word embedding object, OR graph object
    '''
    net = nx.DiGraph()
    visual_net = Network(notebook=True)
    if isinstance(obj,str):
        net = build_graph(build_file(obj))
    elif isinstance(obj,list):
        net = build_graph(build_word_embedding(build_words(obj)))
    elif isinstance(obj,WordEmbeddingBase):
        net = build_graph(build_word_embedding(obj))
    elif isinstance(obj,GraphBase):
        net = build_graph(obj)
    else:
        raise TypeError('Please provide a file name, words list, word embedding, or graph.')
    if scaled:
        in_dict = dict(net.in_degree)
        min_size = 5
        scale = 10
        in_dict.update((word, min_size + scale*in_degree) for word,in_degree in in_dict.items())
        nx.set_node_attributes(net, in_dict, 'size')
    else:
        nx.set_node_attributes(net, 'circle', 'shape')
    visual_net.from_nx(net)
    visual_net.show(file_name + '.html')

def build_graph(graph): # graph object => nx.Digraph object
    net = nx.DiGraph()
    for i in range(graph.number_of_vertices()):
        net.add_node(graph.get_vertex(i).word)
    for i in range(graph.number_of_edges()):
        edge = graph.get_edge(i)
        net.add_edge(edge.vertex_out, edge.vertex_in, arrows="to")
    return net

def build_word_embedding(word_embedding): # word_embedding object => graph object
    graph_txt_1 = graph_txt.GraphTXT()
    topics_utils.generate_topics_graph(word_embedding, graph_txt_1)
    graph_txt_1.graph_to_file()
    return graph_txt_1

def build_words(words): # string list => word_embedding object
    words = list(set(words))
    # Load the pre-trained word embeddings
    glove = torchtext.vocab.GloVe(name='6B', dim=50)
    word_embedding_basic_1 = WordEmbeddingBasic(50)
    words_kept = 0
    for word in words:
        word = word.lower()
        if word in glove.stoi:
            word_embedding = glove.vectors[glove.stoi[word.lower()]]
            word_embedding_list = word_embedding.numpy().tolist()
            word_embedding_basic_1.add_word_vector((word, word_embedding_list))
            words_kept += 1
    print("Number of words kept: " + str(words_kept))
    return word_embedding_basic_1

def build_file(file_name):
    return graph_txt.GraphTXT(file_name)

