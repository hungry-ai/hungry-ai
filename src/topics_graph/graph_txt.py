from enum import Enum
from .graph_base import GraphBase

class VertexType(Enum):
    USER = 0
    IMAGE = 1
    TOPIC = 2

class VertexTXT():
    def __init__(self, vertex_type, word, word_vector, id = None):
        self.type = vertex_type
        # If the graph type isn't a topic, the word is just image_743
        # or user_8294. Whatever the id is. 
        if self.type == VertexType.USER:
            self.word = "user_" + str(id)
        elif self.type == VertexType.IMAGE:
            self.word = "image_" + str(id)
        elif self.type == VertexType.TOPIC:
            self.word = word
        else:
            raise TypeError("Input vertex_type = " + str(self.type) + " is not one of USER, IMAGE, TOPIC.")
        self.word_vector = word_vector
        self.id = id

class EdgeTXT():
    def __init__(self, vertex_out, vertex_in, edge_weight, id = None):
        self.vertex_out = vertex_out
        self.vertex_in = vertex_in
        self.edge_weight = edge_weight
        self.id = id

class GraphTXT(GraphBase):
    def __init__(self, file_name = None):
        self.file_name = file_name
        # List of vertices of graph. self.vertex_list[k] is the vertex with id k.
        self.vertex_list = []
        # List of edges of graph. self.edge_list[k] is the edge with id k.
        self.edge_list = []
        # Keys are words. self.vertex_dict[word] is a VertexTXT object, v, 
        # such that v.word = word.
        self.vertex_dict = dict()
        # Keys are words. self.neighbors_lists[word] is a list of vertex_id's such 
        # that there is an edge from word to each vertex_id.
        self.neighbors_lists = dict()
        # Keys are words. self.neighbors_dicts[word] is a dictionary where keys are 
        # words. self.neighbors_dicts[word_1][word_2] exists iff there is a edge from
        # word_1 to word_2 and the value is the edge_id of said edge.
        self.neighbors_dicts = dict()
        self.number_edges = 0
        self.number_vertices = 0
        if file_name != None:
            self.load_graph()
    
    # Loads a graph from csv file.
    def load_graph(self):
        with open(self.file_name, 'r') as file:
            line_list = file.readline().split(' ')
            if len(line_list)!= 2:
                raise ValueError("CSV formatted incorrectly! First line must contain 2 items!")
            vertex_count, edge_count = line_list
            vertex_count = int(vertex_count)
            edge_count = int(edge_count)
            for i in range(vertex_count):
                if self.add_vertex_from_string(file.readline()) == None:
                    raise ValueError("Failed attempting to load vertex. Line number in file: " + str(1 + i))
            for i in range(edge_count):
                if self.add_edge_from_string(file.readline()) == None:
                    raise ValueError("Failed attempting to load edge. Line number in file: " + str(1 + vertex_count + i))

    # Returns number of vertices of the graph.
    def number_of_vertices(self):
        return self.number_vertices

    # Returns number of edges of the graph.
    def number_of_edges(self):
        return self.number_edges

    # Returns the i-th vertex of the graph.
    # This function is used to iterate through all vertices.
    def get_vertex(self, i):
        if 0 <= i and i < self.number_vertices:  
            return self.vertex_list[i]
        return None

    # Returns the vertex with the input name.
    def get_vertex_with_name(self, name):
        if name in self.vertex_dict:
            vertex_id = self.vertex_dict[name]
            return self.vertex_list[vertex_id]
        return None

    def get_edge(self, i):
        if 0 <= i and i < self.number_edges:
            return self.edge_list[i]
        return None

    # Returns the edge (v,u) where v and u are vertex IDs. Returns None if there is no edge.
    def get_edge_with_name(self, v, u):
        if v in self.neighbors_dicts and u in self.neighbors_dicts[v]:
            edge_id = self.neighbors_dicts[v][u]
            return self.edge_list[edge_id]
        return None
    
    # Returns the i-th neighbor of the vertex v.
    # This function is used to iterate through all neighbors of a vertex.
    def get_neighbor(self, v, i):
        if v in self.neighbors_lists and 0 <= i and i < len(self.neighbors_lists[v]):
            neighbor_word = self.neighbors_lists[v][i]
            return self.vertex_dict[neighbor_word]
        return None
    
    def add_vertex(self, vertex):
        '''
        vertex: list containing (optional: vertex_type), word, word_vector
        '''
        if len(vertex) == 2:
            word, word_vector = vertex
            vertex_type = VertexType.TOPIC
        else:
            vertex_type, word, word_vector = vertex
            vertex_type = VertexType(int(vertex_type))
        vertex_id = self.number_vertices
        vertex_txt = VertexTXT(vertex_type, word, word_vector, vertex_id)
        word = vertex_txt.word
        self.vertex_list.append(vertex_txt)
        self.vertex_dict[word] = vertex_id
        self.neighbors_lists[word] = []
        self.neighbors_dicts[word] = dict()
        self.number_vertices += 1
        return self.number_vertices

    def add_edge(self, edge):
        '''
        edge: list containing vertex_in, vertex_out, edge_weight
        '''
        vertex_in, vertex_out, edge_weight = edge
        edge_id = self.number_edges
        edge_txt = EdgeTXT(vertex_in, vertex_out, edge_weight, edge_id)
        self.edge_list.append(edge_txt)
        self.neighbors_lists[vertex_in].append(vertex_out)
        self.neighbors_dicts[vertex_in][vertex_out] = edge_id
        self.number_edges += 1
        return self.number_edges

    # Adds a vertex to the graph.
    def add_vertex_from_string(self, vertex_string):
        vertex_items = vertex_string.split(' ', maxsplit = 2)
        if len(vertex_items) != 3:
            print("CSV file format incorrect! Current line is not correct format for vertex!")
            return None
        vertex_type, word, word_vector_string = vertex_items
        # Get rid of the brackets []
        word_vector_string = word_vector_string[1: len(word_vector_string)-1]
        word_vector = word_vector_string.split(',')
        return self.add_vertex((vertex_type, word, word_vector))

    # Adds an edge to the graph.
    def add_edge_from_string(self, edge_string):
        edge_items = edge_string.split(' ')
        if len(edge_items) != 3:
            print("CSV file format incorrect! Current line is not correct format for edge!")
            return None
        vertex_in, vertex_out, edge_weight = edge_items
        edge_weight = float(edge_weight)
        return self.add_edge((vertex_in, vertex_out, edge_weight))
    
    def graph_to_file(self, filename = None):
        if filename == None:
            filename = "sample.txt"
        with open(filename, "w") as file:
            file.write(str(self.number_of_vertices()) + " " + str(self.number_of_edges()) + "\n")
            for i in range(self.number_of_vertices()):
                vertex = self.get_vertex(i)
                file.write(str(vertex.type.value) + " " + vertex.word + " " + str(vertex.word_vector) + "\n")
            for i in range(self.number_of_edges()):
                edge = self.edge_list[i]
                file.write(edge.vertex_out + " " + edge.vertex_in + " " + str(edge.edge_weight) + "\n")
        return filename
