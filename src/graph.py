import datetime
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np

from .db import Edge, EdgeDB, Review


class VertexType(Enum):
    USER = 0
    IMAGE = 1
    TOPIC = 2


@dataclass(frozen=True)
class Vertex:
    id: str
    type: int


class GraphService:
    def __init__(
        self,
        edge_db: EdgeDB,
        rating_halflife: datetime.timedelta = datetime.timedelta(
            days=365
        ),  # TODO: choose optimal halflife?
    ) -> None:
        self.edge_db = edge_db

        self.in_neighbors: dict[Vertex, dict[Vertex, Edge]] = defaultdict(dict)
        self.out_neighbors: dict[Vertex, dict[Vertex, Edge]] = defaultdict(dict)

        self.rating_halflife = rating_halflife

    def _read_graph(self) -> None:
        pass

    def _write_graph(self) -> None:
        pass

    def _add_edge(
        self,
        from_vtx: Vertex,
        to_vtx: Vertex,
        weight: float,
        timestamp: datetime.datetime = datetime.datetime.now(),
    ) -> None:
        edge = Edge(
            from_vtx.id,
            from_vtx.type,
            to_vtx.id,
            to_vtx.type,
            weight,
            timestamp,
        )

        self.out_neighbors[from_vtx][to_vtx] = edge
        self.in_neighbors[to_vtx][from_vtx] = edge

        self.edge_db.insert(edge)

    def add_image_edge(self, image_id: str, topic_id: str, p: float) -> None:  # TODO
        if p < 0.0 or p > 1.0:
            raise ValueError("invalid p")

        from_vtx = Vertex(image_id, VertexType.IMAGE.value)
        to_vtx = Vertex(topic_id, VertexType.TOPIC.value)
        weight = 0.5
        self._add_edge(from_vtx, to_vtx, weight)

    def add_user_edge(self, user_id: str, image_id: str, rating: float) -> None:  # TODO
        if rating < 1.0 or rating > 5.0:
            raise ValueError("invalid rating")

        from_vtx = Vertex(user_id, VertexType.USER.value)
        to_vtx = Vertex(image_id, VertexType.IMAGE.value)
        weight = rating
        timestamp = datetime.datetime.now()
        if to_vtx in self.out_neighbors[from_vtx]:
            edge = self.out_neighbors[from_vtx][to_vtx]
            alpha = np.exp2(-(timestamp - edge.timestamp) / self.rating_halflife)
            weight = weight * (1.0 - alpha) + edge.weight * alpha

        if to_vtx not in self.out_neighbors:
            raise ValueError("unknown image_id")

        self._add_edge(from_vtx, to_vtx, weight, timestamp=timestamp)

    def predict_ratings(self, user_id: str) -> dict[str, float]:  # TODO
        return {
            vertex.id: 2.5
            for vertex in self.out_neighbors
            if vertex.type == VertexType.IMAGE.value
        }
