from graph_base import GraphBase

class GraphBasic(GraphBase):
    def __init__(self):
        # Adjacency list implementation of graph.
        # self.graph[v][k] = (u,w) where u is the k-th outgoing neighbor v
        # and w is the weight of the outgoing edge.
        self.graph = []
        # Basically matrix representation of graph. 
        # self.graph_dict[v][u] is the weight of the edge (v,u) if it exists.
        self.graph_dict = []
        self.num_edges = 0
    
    # Returns number of vertices of the graph.
    def number_of_vertices(self):
        return len(self.graph)

    # Returns number of edges of the graph.
    def number_of_edges(self):
        return self.num_edges
    
    # Returns the i-th vertex of the graph.
    # This function is used to iterate through all vertices.
    def get_vertex(self, i):
        return i

    # Returns the vertex with the input id.
    def get_vertex_with_id(self, id):
        return id

    # Returns the i-th neighbor of the vertex v.
    # This function is used to iterate through all neighbors of a vertex.
    def get_neighbor(self, v, i):
        if i >= len(self.graph[v]): return None
        return self.graph[v][i]

    # Returns the edge (v,u) where v and u are vertex IDs. Returns None if there is no edge.
    def get_edge(self, v, u):
        if u not in self.graph_dict[v]: return None
        return self.graph_dict[v][u]

    # Adds a vertex to the graph.
    def add_vertex(self, v):
        word, vector = v
        self.graph.append([])
        self.graph_dict.append(dict())
        return len(self.graph) - 1

    # Adds an edge to the graph.
    def add_edge(self, edge):
        (v,u,w) = edge
        if v >= len(self.graph): return None
        self.graph[v].append((u,w))
        self.graph_dict[v][u] = w
        self.num_edges += 1
        return edge
