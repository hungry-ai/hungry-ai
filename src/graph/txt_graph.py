from .graph import Graph
from .vertex import Vertex, VertexType
import heapq
import pandas as pd  # type: ignore[import]
from pathlib import Path


class TXTGraph(Graph):
    def __init__(self, root: Path) -> None:
        super().__init__()

        self.root = root
        self.vertices_path = root / "vertices.csv"
        self.edges_path = root / "edges.csv"

        if self.vertices_path.exists() and self.edges_path.exists():
            self.read_graph()

    def read_graph(self):
        vertices_df = pd.read_csv(self.vertices_path)
        vertices = dict()
        for _, row in vertices_df.iterrows():
            vertex = Vertex(row["id"], row["type"], row["label"])
            vertices[vertex.id] = vertex
            self.add_vertex(vertex)

        edges_df = pd.read_csv(self.edges_path)
        for _, row in edges_df.iterrows():
            self.add_edge(
                vertices[row["src_id"]], vertices[row["dest_id"]], row["weight"]
            )

    def write_graph(self) -> None:
        vertices_df = pd.DataFrame(
            [
                {
                    "id": vertex.id,
                    "type": vertex.type,
                    "label": vertex.label,
                }
                for vertex in self.vertices
            ]
        )
        vertices_df.to_csv(self.vertices_path)

        edges_df = pd.DataFrame(
            [
                {"src_id": src.id, "dest_id": dest.id, "weight": weight}
                for src, dest, weight in self.edges
            ]
        )
        edges_df.to_csv(self.edges_path)
