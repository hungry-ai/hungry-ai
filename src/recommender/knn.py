import heapq
import warnings

import pandas as pd  # type: ignore[import]

from ..graph import Graph, Vertex, VertexType
from ..images import Image, pr_match
from ..reviews import Review
from ..tags import Tag, WordEmbedding
from ..users import User
from .recommender import Recommender


class KNNRecommender(Recommender):
    def __init__(self, graph: Graph, tags: list[Tag]) -> None:
        self.graph = graph
        self.tags = tags
        self.user_image_edges: dict[Vertex, dict[Vertex, Review]] = dict()

    def add_user(self, user: User) -> None:
        if Vertex(user.id, VertexType.USER) in self.graph.vertices:
            warnings.warn(f"{user.id=} already exists in the graph")

        user_vtx = self.graph.add_user(user.id)
        self.user_image_edges[user_vtx] = dict()

    def get_image_tag_weights(self, image: Image) -> dict[Tag, float]:
        return {tag: pr_match(image.url, tag.name) for tag in self.tags}

    def add_image_edge(self, image_id: str, tag_id: str, p: float) -> None:
        if p < 0.0 or p > 1.0:
            raise ValueError("invalid p")

        image_vtx = Vertex(image_id, VertexType.IMAGE)
        if image_vtx not in self.graph.vertices:
            raise KeyError(
                f"unknown {image_id=}, available: {[vertex.id for vertex in self.graph.images]}"
            )

        tag_vtx = Vertex(tag_id, VertexType.TAG)
        if tag_vtx not in self.graph.vertices:
            raise KeyError(
                f"unknown {tag_id=}, available: {[vertex.id for vertex in self.graph.tags]}"
            )

        self.graph.add_edge(image_vtx, tag_vtx, weight=p, directed=False)

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

        user_vtx = Vertex(user_id, VertexType.USER)
        if user_vtx not in self.graph.vertices:
            raise KeyError(
                f"unknown {user_id=}, available: {[vertex.id for vertex in self.graph.users]}"
            )

        image_vtx = Vertex(image_id, VertexType.IMAGE)
        if image_vtx not in self.graph.vertices:
            raise KeyError(
                f"unknown {image_id=}, available: {[vertex.id for vertex in self.graph.images]}"
            )

        self.user_image_edges[user_vtx][image_vtx] = review
        self.update_user_preferences(review.user)

    def update_user_preferences(self, user):
        user_vtx = Vertex(user.id, VertexType.USER)
        total_interest = 0.0
        interest_scores: dict[Vertex, float] = dict()

        for image_vtx in self.user_image_edges[user_vtx]:
            review = self.user_image_edges[user_vtx][image_vtx]
            for topic_vtx in self.graph.out_neighbors(image_vtx):
                interest = self.graph.out_neighbors(image_vtx)[topic_vtx] * review.rating
                total_interest += interest
                if topic_vtx not in interest_scores:
                    interest_scores[topic_vtx] = 0.0
                interest_scores[topic_vtx] += interest
                
        # Update edge weights with 1.0 - normalized interest scores.
        for topic_vtx in self.graph.out_neighbors(image_vtx):
            self.graph.add_edge(user_vtx, topic_vtx, 1.0 - interest_scores[topic_vtx] / total_interest)

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


def train_knn(
    train_data: pd.DataFrame, word_embedding: WordEmbedding
) -> KNNRecommender:
    graph = LocalGraph()
    generate_tags_graph(word_embedding, graph)

    recommender = KNNRecommender(graph)

    return recommender
