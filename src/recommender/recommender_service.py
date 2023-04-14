from __future__ import annotations

from ..db import RecommendationDB
from ..graph import GraphService


class RecommenderService:
    def __init__(
        self,
        recommendation_db: RecommendationDB,
        graph_service: GraphService,
    ) -> None:
        self.recommendation_db = recommendation_db
        self.graph_service = graph_service

        self.rated: set[str] = set()

    def predict_ratings(self, user_id: str) -> dict[str, float]:
        raise NotImplementedError

    def recommend(self, user_id: str) -> None | str:
        ratings = self.predict_ratings(user_id)
        if len(self.rated) == len(ratings):
            self.rated.clear()

        for rating, image_id in sorted(
            [(v, k) for k, v in ratings.items()], reverse=True
        ):
            if image_id in self.rated:
                continue
            return image_id

        return None


# TODO: incorporate this


"""
def predict_ratings(self, user_id: str) -> dict[str, float]:  # TODO
    return {vertex.id: 2.5 for vertex in self.graph.images}

def get_recommendations(
    graph: GraphService, username: str, number_of_recommendations: int
) -> tuple[list[tuple[str, float]], dict[Vertex, tuple[float, Vertex | None]]]:
    src = graph.users[username]

    distances: dict[Vertex, tuple[float, Vertex | None]] = dict()
    for vertex in graph.vertices:
        distances[vertex] = (float("infinity"), None)

    distances[src] = (0.0, None)
    priority_queue: list[tuple[float, Vertex, Vertex | None]] = [(0.0, src, None)]
    recommended_images = []

    while priority_queue:
        current_distance, current_vertex, previous = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex][0]:
            continue

        if current_vertex.type == VertexType.IMAGE:
            recommended_images.append((current_vertex.label, current_distance))

        for neighbor, weight in graph.out_neighbors[current_vertex].items():
            distance = current_distance + weight

            if distance < distances[neighbor][0]:
                distances[neighbor] = (distance, current_vertex)
                heapq.heappush(priority_queue, (distance, neighbor, current_vertex))

    return (recommended_images, distances)
"""
