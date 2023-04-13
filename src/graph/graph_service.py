import datetime

import numpy as np

from .vertex import Vertex, VertexType
from .graph import Graph


class GraphService:
    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def add_user(self, user_id: str) -> None:
        self.graph.add_user(user_id)

    def add_image(self, image_id: str) -> None:
        self.graph.add_image(image_id)

    def add_image_edge(self, image_id: str, tag_id: str, p: float) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError("invalid p")

        from_vtx = Vertex(image_id, VertexType.IMAGE)
        if from_vtx not in self.graph.vertices:
            raise KeyError(
                f"unknown {image_id=}, available: {[vertex.id for vertex in self.graph.images]}"
            )

        to_vtx = Vertex(tag_id, VertexType.TAG)
        if to_vtx not in self.graph.vertices:
            raise KeyError(
                f"unknown {tag_id=}, available: {[vertex.id for vertex in self.graph.tags]}"
            )

        self.graph.add_edge(from_vtx, to_vtx, p)

    def add_user_edge(self, user_id: str, image_id: str, rating: float) -> None:
        if rating < 1.0 or rating > 5.0:
            raise ValueError("invalid rating")

        from_vtx = Vertex(user_id, VertexType.USER)
        if from_vtx not in self.graph.vertices:
            raise ValueError(
                f"unknown {user_id=}, available: {[vertex.id for vertex in self.graph.users]}"
            )

        to_vtx = Vertex(image_id, VertexType.IMAGE)
        if to_vtx not in self.graph.vertices:
            raise ValueError(
                f"unknown {image_id=}, available: {[vertex.id for vertex in self.graph.images]}"
            )

        if to_vtx in self.graph.out_neighbors(from_vtx):
            raise NotImplementedError("TODO: support multiple user->image edges")

        self.graph.add_edge(from_vtx, to_vtx, weight=rating)

    def predict_ratings(self, user_id: str) -> dict[str, float]:  # TODO
        return {vertex.id: 2.5 for vertex in self.graph.images}
