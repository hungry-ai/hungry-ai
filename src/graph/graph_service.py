import datetime

import numpy as np

from ..db import Edge, EdgeDB
from .vertex import Vertex, VertexType
from .local_graph import LocalGraph


class GraphService(LocalGraph):
    def __init__(
        self,
        edge_db: EdgeDB,
        rating_halflife: datetime.timedelta = datetime.timedelta(
            days=365
        ),  # TODO: choose optimal halflife?
    ) -> None:
        super().__init__()

        self.edge_db = edge_db
        self.rating_halflife = rating_halflife
        self.edge_timestamps: dict[tuple[Vertex, Vertex], datetime.datetime] = dict()

        self._read_graph()

    def _read_graph(self) -> None:
        edges = self.edge_db.select()

        for edge in edges:
            from_vtx = Vertex(edge.from_id, edge.from_type)
            if from_vtx not in self.vertices:
                self.add_vertex(from_vtx)

            to_vtx = Vertex(edge.to_id, edge.to_type)
            if to_vtx not in self.vertices:
                self.add_vertex(to_vtx)

            self.add_edge(from_vtx, to_vtx, weight=edge.weight)
            self.edge_timestamps[(from_vtx, to_vtx)] = edge.timestamp

    def _add_edge(
        self,
        from_vtx: Vertex,
        to_vtx: Vertex,
        weight: float,
        timestamp: datetime.datetime = datetime.datetime.now(),
    ) -> None:
        self.add_edge(from_vtx, to_vtx, weight=weight)
        self.edge_timestamps[(from_vtx, to_vtx)] = timestamp

        edge = Edge(
            from_vtx.id,
            from_vtx.type,
            to_vtx.id,
            to_vtx.type,
            weight,
            timestamp,
        )
        self.edge_db.insert(edge)

    def add_image_edge(self, image_id: str, tag_id: str, p: float) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError("invalid p")

        from_vtx = Vertex(image_id, VertexType.IMAGE.value)
        if from_vtx not in self.vertices:
            self.add_vertex(from_vtx)

        to_vtx = Vertex(tag_id, VertexType.TAG.value)
        if to_vtx not in self.vertices:
            self.add_vertex(to_vtx)

        self._add_edge(from_vtx, to_vtx, p)

    def add_user_edge(self, user_id: str, image_id: str, rating: float) -> None:
        if rating < 1.0 or rating > 5.0:
            raise ValueError("invalid rating")

        to_vtx = Vertex(image_id, VertexType.IMAGE.value)
        if to_vtx not in self.vertices:
            raise ValueError("unknown image_id")

        from_vtx = Vertex(user_id, VertexType.USER.value)
        if from_vtx not in self.vertices:
            self.add_vertex(from_vtx)

        weight = rating
        timestamp = datetime.datetime.now()

        if to_vtx in self.out_neighbors(from_vtx):
            current_weight = self.out_neighbors(from_vtx)[to_vtx]
            current_timestamp = self.edge_timestamps[(from_vtx, to_vtx)]
            alpha = np.exp2(-(timestamp - current_timestamp) / self.rating_halflife)
            weight = weight * (1.0 - alpha) + current_weight * alpha

        self._add_edge(from_vtx, to_vtx, weight, timestamp=timestamp)

    def predict_ratings(self, user_id: str) -> dict[str, float]:  # TODO
        return {
            vertex.id: 2.5
            for vertex in self.vertices
            if vertex.type == VertexType.IMAGE.value
        }
