from .db import Graph, GraphDB, Review


class GraphService:
    def __init__(self, graph_db: GraphDB) -> None:
        self.graph_db = graph_db

    def add_image_edge(self, image_id: str, topic_id: str, p: float) -> None:  # TODO
        if p < 0.0 or p > 1.0:
            raise ValueError("invalid p")

        pass

    def add_user_edge(self, user_id: str, image_id: str, rating: float) -> None:  # TODO
        if rating < 1.0 or rating > 5.0:
            raise ValueError("invalid rating")

        # if image_id not seen before:
        #     raise ValueError("unknown image_id")

        pass

    def predict_ratings(self, user_id: str) -> dict[str, float]:  # TODO
        return {}
