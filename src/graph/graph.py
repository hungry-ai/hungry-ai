from abc import ABCMeta, abstractmethod
from .vertex import Vertex, VertexType


class Graph(metaclass=ABCMeta):
    @abstractmethod
    def add_vertex(self, vertex: Vertex) -> None:
        raise NotImplementedError

    def add_tag(self, tag_id: str) -> Vertex:
        vertex = Vertex(tag_id, VertexType.TAG)
        self.add_vertex(vertex)
        return vertex

    def add_user(self, user_id: str) -> Vertex:
        vertex = Vertex(user_id, VertexType.USER)
        self.add_vertex(vertex)
        return vertex

    def add_image(self, image_id: str) -> Vertex:
        vertex = Vertex(image_id, VertexType.IMAGE)
        self.add_vertex(vertex)
        return vertex

    @property
    @abstractmethod
    def vertices(self) -> set[Vertex]:
        raise NotImplementedError

    @property
    def tags(self) -> set[Vertex]:
        return {vertex for vertex in self.vertices if vertex.type == VertexType.TAG}

    @property
    def users(self) -> set[Vertex]:
        return {vertex for vertex in self.vertices if vertex.type == VertexType.USER}

    @property
    def images(self) -> set[Vertex]:
        return {vertex for vertex in self.vertices if vertex.type == VertexType.IMAGE}

    @abstractmethod
    def add_directed_edge(self, src: Vertex, dest: Vertex, weight: float = 1.0) -> None:
        raise NotImplementedError

    def add_edge(
        self, src: Vertex, dest: Vertex, weight: float = 1.0, directed: bool = True
    ) -> None:
        self.add_directed_edge(src, dest, weight=weight)

        if not directed:
            self.add_directed_edge(dest, src, weight=weight)

    @abstractmethod
    def out_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        raise NotImplementedError

    @abstractmethod
    def in_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        raise NotImplementedError
