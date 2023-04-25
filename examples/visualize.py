from pathlib import Path

from src.graph import LocalGraph, visualize, build_path

def small_example():
    graph1 = LocalGraph()
    vertex1 = graph1.add_tag("0")
    vertex2 = graph1.add_image("1")
    vertex3 = graph1.add_user("2")
    graph1.add_edge(vertex1, vertex2, 3.0)
    graph1.add_edge(vertex2, vertex3, 2.0)
    labels = {vertex1: "Japanese Food", vertex2: "Ramen Image", vertex3: "Zozo"}
    path = set([vertex1, vertex2])

    visualize(graph1, labels, Path("types_graph.html"), path=path)
    

def path_example():
    graph = LocalGraph()

    soup = graph.add_tag("soup")
    ramen = graph.add_tag("ramen")
    japanese = graph.add_tag("japanese")

    graph.add_edge(ramen, soup)
    graph.add_edge(ramen, japanese)

    tonkotsu = graph.add_image("tonkotsu")
    chicken_noodle = graph.add_image("chicken_noodle")

    graph.add_edge(tonkotsu, soup, 1.0, directed=False)
    graph.add_edge(tonkotsu, ramen, 1.0, directed=False)
    graph.add_edge(tonkotsu, japanese, 1.0, directed=False)
    graph.add_edge(chicken_noodle, soup, 1.0, directed=False)

    cody = graph.add_user("cody")
    alex = graph.add_user("alex")

    graph.add_edge(cody, tonkotsu, 5.0, directed=False)
    graph.add_edge(alex, tonkotsu, 5.0, directed=False)
    graph.add_edge(alex, chicken_noodle, 1.0, directed=False)
    
    labels2 = {soup:"soup", ramen:"ramen", japanese:"japanese",
               tonkotsu:"tonkotsu", chicken_noodle:"chicken_noodle", cody:"cody", alex:"alex"}
    path2 = {chicken_noodle:soup, soup:ramen, alex:chicken_noodle}
    path2 = build_path(path2)
    visualize(graph,labels=labels2, file_name=Path("path_graph.html"), path=path2)

path_example()

