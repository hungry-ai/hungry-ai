from enum import Enum
from .graph_base import GraphBase
import heapq

class VertexType(Enum):
    USER = 0
    IMAGE = 1
    TOPIC = 2

class VertexTXT():
    def __init__(self, vertex_type, name, value, id = None):
        self.type = vertex_type
        # If the graph type isn't a topic, the word is just image_743
        # or user_8294. Whatever the id is. 
        if self.type == VertexType.USER:
            self.name = "user_" + str(id)
        elif self.type == VertexType.IMAGE:
            self.name = "image_" + str(id)
        elif self.type == VertexType.TOPIC:
            self.name = name
        else:
            raise TypeError("Input vertex_type = " + str(self.type) + " is not one of USER, IMAGE, TOPIC.")
        self.value = value
        self.id = id

class EdgeTXT():
    def __init__(self, vertex_out, vertex_in, edge_weight, id = None):
        # The edge is from vertex_out to vertex_in.
        # In other words, vertex_out is the source and vertex_in is the sink.
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
        # such that v.name = word.
        self.vertex_dict = dict()
        # Keys are words. self.edge_dict[word] is a dictionary where keys are 
        # words. self.edge_dict[word_1][word_2] exists iff there is a edge from
        # word_1 to word_2 and the value is the edge_id of said edge.
        self.edge_dict = dict()
        # Keys are words. self.out_neighbors_lists[word] is a list of vertex_id's such 
        # that there is an edge from word to each vertex_id.
        self.out_neighbors_lists = dict()
        # Keys are words. self.in_neighbors_lists[word] is a list of vertex_id's such 
        # that there is an edge from vertex_id to word.
        self.in_neighbors_lists = dict()
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
    
    # Returns number of edges coming into v.
    def get_in_degree(self, v):
        return len(self.in_neighbors_lists[v])

    # Returns number of edges going out of v.
    def get_out_degree(self, v):
        return len(self.out_neighbors_lists[v])

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
    
    # Returns i-th edge of the graph.
    # This functon is used to iterate through all edges.
    def get_edge(self, i):
        if 0 <= i and i < self.number_edges:
            return self.edge_list[i]
        return None

    # Returns the edge (v,u) where v and u are vertex names or words. Returns None if there is no edge.
    def get_edge_with_name(self, v, u):
        if v in self.edge_dict and u in self.edge_dict[v]:
            edge_id = self.edge_dict[v][u]
            return self.edge_list[edge_id]
        return None
    
    # Returns the i-th outgoing neighbor, call it u, of the vertex v.
    # Outgoing as in, there is an edge from v to u.
    # This function is used to iterate through all neighbors of a vertex.
    def get_out_neighbor(self, v, i):
        if v in self.out_neighbors_lists and 0 <= i and i < len(self.out_neighbors_lists[v]):
            neighbor_word = self.out_neighbors_lists[v][i]
            return self.vertex_dict[neighbor_word]
        return None
    
    # Returns the i-th incoming neighbor, call it u, of the vertex v.
    # Incoming as in, there is an edge from u to v.
    # This function is used to iterate through all neighbors of a vertex.
    def get_in_neighbor(self, v, i):
        if v in self.in_neighbors_lists and 0 <= i and i < len(self.in_neighbors_lists[v]):
            neighbor_word = self.in_neighbors_lists[v][i]
            return self.vertex_dict[neighbor_word]
        return None
    
    def add_topic_vertex(self, vertex):
        return self.add_vertex(vertex)
    
    def add_user_vertex(self, vertex):
        pass

    def add_image_vertex(self, vertex):
        pass
    
    def add_vertex(self, vertex):
        '''
        vertex: list containing (optional: vertex_type), word, word_vector
        '''
        if len(vertex) == 2:
            name, value = vertex
            vertex_type = VertexType.TOPIC
        elif len(vertex) == 3:
            vertex_type, name, value = vertex
            vertex_type = VertexType(int(vertex_type))
            if vertex_type == VertexType.USER or vertex_type == VertexType.IMAGE:
                value = name
        else: 
            raise TypeError("Add vertex failed! Not enough elements to parse!")
        vertex_id = self.number_vertices
        vertex_txt = VertexTXT(vertex_type, name, value, vertex_id)
        name = vertex_txt.name
        self.vertex_list.append(vertex_txt)
        self.vertex_dict[name] = vertex_id
        self.edge_dict[name] = dict()
        self.out_neighbors_lists[name] = []
        self.in_neighbors_lists[name] = []
        self.number_vertices += 1
        return vertex_id

    # Adds a directed edge to the graph.
    def add_edge(self, edge):
        '''
        edge: List (or tuple) containing vertex_out, vertex_in, edge_weight
              The edge is from vertex_out to vertex_in, written as (vertex_out, vertex_in).
        '''
        vertex_out, vertex_in, edge_weight = edge
        if vertex_in not in self.vertex_dict and vertex_out not in self.vertex_dict:
            raise ValueError("Edge can't be added because vertices don't exist!")
        edge_id = self.number_edges
        edge_txt = EdgeTXT(vertex_out, vertex_in, edge_weight, edge_id)
        self.edge_list.append(edge_txt)
        self.out_neighbors_lists[vertex_out].append(vertex_in)
        self.in_neighbors_lists[vertex_in].append(vertex_out)
        self.edge_dict[vertex_out][vertex_in] = edge_id
        self.number_edges += 1
        return edge_id

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
        vertex_out, vertex_in, edge_weight = edge_items
        edge_weight = float(edge_weight)
        return self.add_edge((vertex_out, vertex_in, edge_weight))
    
    def graph_to_file(self, filename = None):
        if filename == None:
            filename = "sample.txt"
        with open(filename, "w") as file:
            file.write(str(self.number_of_vertices()) + " " + str(self.number_of_edges()) + "\n")
            for i in range(self.number_of_vertices()):
                vertex = self.get_vertex(i)
                file.write(str(vertex.type.value) + " " + vertex.name + " " + str(vertex.value) + "\n")
            for i in range(self.number_of_edges()):
                edge = self.edge_list[i]
                file.write(edge.vertex_out + " " + edge.vertex_in + " " + str(edge.edge_weight) + "\n")
        return filename

    def get_recommendations(self, username, number_of_recommendations):
        distances = dict()
        for i in range(self.number_of_vertices()):
            vertex = self.get_vertex(i)
            distances[vertex.name] = (float('infinity'), None)
        
        distances[username] = (0.0, None)
        priority_queue = [(0.0, username, None)]
        recommended_images = []

        while priority_queue:
            current_distance, current_vertex, previous = heapq.heappop(priority_queue)
            if current_distance > distances[current_vertex][0]:
                continue

            if self.vertex_dict[current_vertex].type == VertexType.IMAGE:
                recommended_images.push_back((current_vertex, current_distance))

            for neighbor in self.out_neighbors_lists[current_vertex]:
                distance = current_distance + self.get_edge_with_name(current_vertex, neighbor).edge_weight
                
                if distance < distances[neighbor][0]:
                    distances[neighbor] = (distance, current_vertex)
                    heapq.heappush(priority_queue, (distance, neighbor, current_vertex))

        return (recommended_images, distances)
