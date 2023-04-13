from __future__ import annotations


from abc import ABCMeta, abstractmethod
from uuid import uuid4
from .vertex import Vertex, VertexType


class Graph:
    def __init__(self) -> None:
        self.vertices: set[Vertex] = set()
        self.tags: set[Vertex] = set()
        self.users: set[Vertex] = set()
        self.images: set[Vertex] = set()

        self.labels: dict[Vertex, str] = dict()

        self.out_neighbors: dict[Vertex, dict[Vertex, float]] = dict()
        self.in_neighbors: dict[Vertex, dict[Vertex, float]] = dict()

    def add_vertex(self, vertex: Vertex, label: str = "") -> None:
        self.vertices.add(vertex)
        self.out_neighbors[vertex] = dict()
        self.in_neighbors[vertex] = dict()
        self.labels[vertex] = label

    def add_tag(
        self, vertex: Vertex = Vertex(str(uuid4()), VertexType.TAG), label: str = ""
    ) -> Vertex:
        self.add_vertex(vertex, label=label)
        self.tags.add(vertex)
        return vertex

    def add_user(
        self, vertex: Vertex = Vertex(str(uuid4()), VertexType.USER), label: str = ""
    ) -> Vertex:
        self.add_vertex(vertex, label=label)
        self.users.add(vertex)
        return vertex

    def add_image(
        self, vertex: Vertex = Vertex(str(uuid4()), VertexType.IMAGE), label: str = ""
    ) -> Vertex:
        self.add_vertex(vertex, label=label)
        self.images.add(vertex)
        return vertex

    def add_edge(
        self, src: Vertex, dest: Vertex, weight: float = 1.0, directed: bool = True
    ) -> None:
        self.out_neighbors[src][dest] = weight

        if not directed:
            self.in_neighbors[dest][src] = weight

    @property
    def edges(self) -> list[tuple[Vertex, Vertex, float]]:
        return [
            (src, dest, weight)
            for src, out_neighbors in self.out_neighbors.items()
            for dest, weight in out_neighbors.items()
        ]
