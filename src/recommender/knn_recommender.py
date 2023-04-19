import heapq

from .recommender import Recommender
from ..graph import Graph, Vertex, VertexType


class KNNRecommender(Recommender):
    def __init__(self, graph: Graph) -> None:
        super().__init__(graph)

    def predict_rating(self, user_id: str, image_id: str) -> float:
        return 2.5

    def get_closest_images(self, user_id: str) -> (list[str], dict[Vertex, Vertex]):
         # dijkstra
        src = Vertex(user_id, VertexType.USER)
        if src not in self.graph.vertices:
            return ([], dict())

        recommendations = []
        distance = {src: 0.0}
        parent = dict()

        priority_queue = []
        heapq.heappush(priority_queue, (distance[src], src.id, src.type, src, None))

        while len(priority_queue):
            _, _, _, current_vertex, previous_vertex = heapq.heappop(priority_queue)
            if current_vertex in parent:
                continue
            parent[current_vertex] = previous_vertex
            for neighbor, weight in self.graph.out_neighbors(current_vertex).items():
                if (
                    neighbor not in distance
                    or distance[current_vertex] + weight < distance[neighbor]
                ):
                    distance[neighbor] = distance[current_vertex] + weight
                    heapq.heappush(
                        priority_queue,
                        (-distance[neighbor], neighbor.id, neighbor.type, neighbor, current_vertex),
                    )
            if current_vertex.type == VertexType.IMAGE:
                recommendations.append(current_vertex.id)
        
        return (recommendations, parent)

    def get_recommendations(self, user_id: str, num_recs: int) -> list[str]:
        closest_images, parent = self.get_closest_images(user_id)
        return closest_images[:num_recs]