from pathlib import Path

from src.graph import LocalGraph, visualize

graph1 = LocalGraph()
vertex1 = graph1.add_tag("0")
vertex2 = graph1.add_image("1")
vertex3 = graph1.add_user("2")
graph1.add_edge(vertex1, vertex2, 3.0)
graph1.add_edge(vertex2, vertex3, 2.0)
labels = {vertex1: "Japanese Food", vertex2: "Ramen Image", vertex3: "Zozo"}

visualize(graph1, labels, Path("types_graph.html"), path=path)
