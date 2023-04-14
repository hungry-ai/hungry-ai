from pathlib import Path

import pandas as pd  # type: ignore[import]

from .graph import Graph
from .vertex import Vertex, VertexType


class CSVGraph(Graph):
    def __init__(self, root: Path) -> None:
        super().__init__()

        self.root = root
        self.vertices_path = root / "vertices.csv"
        self.edges_path = root / "edges.csv"

    def add_vertex(self, vertex: Vertex) -> None:
        vertices, edges = self.read_graph()

        vertices = vertices.append(
            {"id": vertex.id, "type": vertex.type.value}, ignore_index=True
        )

        self.write_graph(vertices, edges)

    @property
    def vertices(self) -> set[Vertex]:
        vertices, edges = self.read_graph()

        return {
            Vertex(row["id"], VertexType(row["type"])) for _, row in vertices.iterrows()
        }

    def add_directed_edge(self, src: Vertex, dest: Vertex, weight: float = 1.0) -> None:
        vertices, edges = self.read_graph()

        edges = edges.append(
            {
                "src_id": src.id,
                "src_type": src.type.value,
                "dest_id": dest.id,
                "dest_type": dest.type.value,
                "weight": weight,
            },
            ignore_index=True,
        )

        self.write_graph(vertices, edges)

    def out_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        if vertex not in self.vertices:
            raise KeyError("{vertex=} not found")

        vertices, edges = self.read_graph()

        return {
            Vertex(row["dest_id"], VertexType(row["dest_type"])): row["weight"]
            for _, row in edges[
                (edges["src_id"] == vertex.id)
                & (edges["src_type"] == vertex.type.value)
            ].iterrows()
        }

    def in_neighbors(self, vertex: Vertex) -> dict[Vertex, float]:
        if vertex not in self.vertices:
            raise KeyError("{vertex=} not found")

        vertices, edges = self.read_graph()

        return {
            Vertex(row["src_id"], VertexType(row["src_type"])): row["weight"]
            for _, row in edges[
                (edges["dest_id"] == vertex.id)
                & (edges["dest_type"] == vertex.type.value)
            ].iterrows()
        }

    def read_graph(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        vertices = (
            pd.read_csv(self.vertices_path, dtype={"id": "str", "type": "int"})
            if self.vertices_path.exists()
            else pd.DataFrame(
                {"id": pd.Series(dtype="str"), "type": pd.Series(dtype="int")}
            )
        )
        edges = (
            pd.read_csv(
                self.edges_path,
                dtype={
                    "src_id": "str",
                    "src_type": "int",
                    "dest_id": "str",
                    "dest_type": "int",
                    "weight": "float",
                },
            )
            if self.edges_path.exists()
            else pd.DataFrame(
                {
                    "src_id": pd.Series(dtype="str"),
                    "src_type": pd.Series(dtype="int"),
                    "dest_id": pd.Series(dtype="str"),
                    "dest_type": pd.Series(dtype="int"),
                    "weight": pd.Series(dtype="float"),
                }
            )
        )
        return vertices, edges

    def write_graph(self, vertices: pd.DataFrame, edges: pd.DataFrame) -> None:
        vertices.to_csv(self.vertices_path, index=False)
        edges.to_csv(self.edges_path, index=False)
