from __future__ import annotations


from abc import ABCMeta, abstractmethod
from uuid import uuid4
from .vertex import Vertex, VertexType


class Graph(metaclass=ABCMeta):
    @abstractmethod
    def add_vertex(self, vertex: Vertex) -> None:
        raise NotImplementedError

    def add_tag(self, vertex: Vertex = Vertex(str(uuid4()), VertexType.TAG)) -> Vertex:
        self.add_vertex(vertex)
        return vertex

    def add_user(
        self, vertex: Vertex = Vertex(str(uuid4()), VertexType.USER)
    ) -> Vertex:
        self.add_vertex(vertex)
        return vertex

    def add_image(
        self, vertex: Vertex = Vertex(str(uuid4()), VertexType.IMAGE)
    ) -> Vertex:
        self.add_vertex(vertex)
        return vertex

    @abstractmethod
    def add_directed_edge(self, src: Vertex, dest: Vertex, weight: float = 1.0) -> None:
        raise NotImplementedError

    def add_edge(
        self, src: Vertex, dest: Vertex, weight: float = 1.0, directed: bool = True
    ) -> None:
        self.add_directed_edge(src, dest, weight=weight)

        if not directed:
            self.add_directed_edge(dest, src, weight=weight)

    @property
    @abstractmethod
    def vertices(self) -> set[Vertex]:
        raise NotImplementedError

    @abstractmethod
    def out_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        raise NotImplementedError

    @abstractmethod
    def in_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        raise NotImplementedError
