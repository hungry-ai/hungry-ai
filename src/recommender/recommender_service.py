from __future__ import annotations

import heapq

from ..db import RecommendationDB
from ..graph import GraphService, Vertex, VertexType


class RecommenderService:
    def __init__(
        self,
        recommendation_db: RecommendationDB,
        graph_service: GraphService,  # TODO: should this have graph_service or graph directly?
    ) -> None:
        self.recommendation_db = recommendation_db
        self.graph_service = graph_service
        self.graph = graph_service.graph

    def predict_rating(self, user_id: str, image_id: str) -> float:
        return 2.5

    def get_recommendations(
        self, user_id: str, number_of_recommendations: int
    ) -> list[str]:
        # dijkstra
        src = Vertex(user_id, VertexType.USER)

        distances: dict[Vertex, tuple[float, Vertex | None]] = dict()
        for vertex in self.graph.vertices:
            distances[vertex] = (float("infinity"), None)

        distances[src] = (0.0, None)
        priority_queue: list[tuple[float, Vertex, Vertex | None]] = [(0.0, src, None)]
        recommended_images = []

        while priority_queue:
            current_distance, current_vertex, previous = heapq.heappop(priority_queue)

            if current_distance > distances[current_vertex][0]:
                continue

            if current_vertex.type == VertexType.IMAGE:
                recommended_images.append(current_vertex.id)

            for neighbor, weight in self.graph.out_neighbors(current_vertex).items():
                distance = current_distance + weight

                if distance < distances[neighbor][0]:
                    distances[neighbor] = (distance, current_vertex)
                    heapq.heappush(priority_queue, (distance, neighbor, current_vertex))

        return recommended_images
