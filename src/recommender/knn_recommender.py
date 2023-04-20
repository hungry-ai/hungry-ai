import heapq
import warnings

from ..graph import Graph, Vertex, VertexType
from ..images import Image, pr_match
from ..reviews import Review
from ..tags import Tag
from ..users import User
from .recommender import Recommender


class KNNRecommender(Recommender):
    def __init__(self, graph: Graph, tags: list[Tag]) -> None:
        self.graph = graph
        self.tags = tags

    def add_user(self, user: User) -> None:
        if Vertex(user.id, VertexType.USER) in self.graph.vertices:
            warnings.warn(f"{user.id=} already exists in the graph")

        self.graph.add_user(user.id)

    def get_image_tag_weights(self, image: Image) -> dict[Tag, float]:
        return {tag: pr_match(image.url, tag.name) for tag in self.tags}

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

        self.graph.add_edge(from_vtx, to_vtx, weight=p, directed=False)

    def add_image(self, image: Image) -> None:
        image_id = image.id

        if Vertex(image_id, VertexType.IMAGE) in self.graph.vertices:
            warnings.warn(f"{image_id=} already exists in the graph")

        self.graph.add_image(image_id)

        for tag, p in self.get_image_tag_weights(image).items():
            self.add_image_edge(image_id, tag.id, p)

    def add_review(self, review: Review) -> None:
        user_id = review.user.id
        image_id = review.image.id
        rating = review.rating

        if rating < 1.0 or rating > 5.0:
            raise ValueError("invalid rating")

        from_vtx = Vertex(user_id, VertexType.USER)
        if from_vtx not in self.graph.vertices:
            raise KeyError(
                f"unknown {user_id=}, available: {[vertex.id for vertex in self.graph.users]}"
            )

        to_vtx = Vertex(image_id, VertexType.IMAGE)
        if to_vtx not in self.graph.vertices:
            raise KeyError(
                f"unknown {image_id=}, available: {[vertex.id for vertex in self.graph.images]}"
            )

        if to_vtx in self.graph.out_neighbors(from_vtx):
            raise NotImplementedError("TODO: support multiple user->image edges")

        self.graph.add_edge(from_vtx, to_vtx, weight=rating, directed=False)

    def get_closest_images(
        self, user_id: str
    ) -> tuple[list[str], dict[Vertex, Vertex | None]]:
        # dijkstra
        src = Vertex(user_id, VertexType.USER)
        if src not in self.graph.vertices:
            return [], dict()

        recommendations = []
        distance = {src: 0.0}
        parent = dict()

        priority_queue: list[tuple[float, str, VertexType, Vertex, Vertex | None]] = []
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
                        (
                            -distance[neighbor],
                            neighbor.id,
                            neighbor.type,
                            neighbor,
                            current_vertex,
                        ),
                    )
            if current_vertex.type == VertexType.IMAGE:
                recommendations.append(current_vertex.id)

        return (recommendations, parent)

    def get_recommendations(self, user: User, num_recs: int) -> list[str]:
        user_id = user.id
        closest_images, parent = self.get_closest_images(user_id)
        return closest_images[:num_recs]
