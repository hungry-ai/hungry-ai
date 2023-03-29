from abc import ABC, abstractmethod

#TODO(gnehzza): Add more comments to these functions.
#TODO(gnehzza): Decide how we want to ID vertices.
class GraphBase(ABC):

    # Returns number of vertices of the graph.
    @abstractmethod
    def number_of_vertices(self):
        pass

    # Returns number of edges of the graph.
    @abstractmethod
    def number_of_edges(self):
        pass
    
    # Returns the i-th vertex of the graph.
    # This function is used to iterate through all vertices.
    @abstractmethod
    def get_vertex(self, i):
        pass

    # Returns the vertex with the input id.
    @abstractmethod
    def get_vertex_with_id(self, id):
        pass

    # Returns the i-th neighbor of the vertex v.
    # This function is used to iterate through all neighbors of a vertex.
    @abstractmethod
    def get_neighbor(self, v, i):
        pass

    # Returns the edge (v,u) where v and u are vertex IDs. Returns None if there is no edge.
    @abstractmethod
    def get_edge(self, v, u):
        pass

    # Adds a vertex to the graph.
    def add_vertex(self, v):
        pass

    # Adds an edge to the graph.
    def add_edge(self, edge):
        pass