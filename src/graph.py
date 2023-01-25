from db import Graph, GraphDB


class GraphService:
    def __init__(self, graph_db: GraphDB) -> None:
        self.graph_db = graph_db

    def add_image_edge(self, image_id: str, topic_id: str, p: float) -> None:  # TODO
        pass

    def add_user_edge(self, user_id: str, image_id: str, rating: int) -> None:  # TODO
        pass

    def predict_ratings(self, user_id: str) -> dict[str, float]:  # TODO
        return {}
