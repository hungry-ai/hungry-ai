import heapq

from .recommender import Recommender
from ..graph import Graph, Vertex, VertexType


class KNNRecommender(Recommender):
    def __init__(self, graph: Graph) -> None:
        super().__init__(graph)

    def predict_rating(self, user_id: str, image_id: str) -> float:
        return 2.5

    def get_recommendations(self, user_id: str, num_recs: int) -> list[str]:
        # dijkstra
        src = Vertex(user_id, VertexType.USER)
        if src not in self.graph.vertices:
            return []

        recommendations = []
        distance = {src: 0.0}
        processed = set()

        priority_queue = []
        heapq.heappush(priority_queue, (distance[src], src.id, src.type, src))

        while len(priority_queue) and len(recommendations) < num_recs:
            _, _, _, current_vertex = heapq.heappop(priority_queue)
            if current_vertex in processed:
                continue
            processed.add(current_vertex)
            for neighbor, weight in self.graph.out_neighbors(current_vertex).items():
                if (
                    neighbor not in distance
                    or distance[current_vertex] + weight < distance[neighbor]
                ):
                    distance[neighbor] = distance[current_vertex] + weight
                    heapq.heappush(
                        priority_queue,
                        (-distance[neighbor], neighbor.id, neighbor.type, neighbor),
                    )
            if current_vertex.type == VertexType.IMAGE:
                recommendations.append(current_vertex.id)
        return recommendations
