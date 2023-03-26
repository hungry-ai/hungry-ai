import datetime

import pytest
from graph_basic import GraphBasic

def test_graph_basic():
    graph_1 = GraphBasic()
    assert 0 == graph_1.number_of_vertices()
    assert 0 == graph_1.number_of_edges()
    
    result = graph_1.add_edge((1,2,1))
    assert None == result
    
    vertex_1 = graph_1.add_vertex()
    vertex_2 = graph_1.add_vertex()
    vertex_3 = graph_1.add_vertex()
    assert 0 == vertex_1
    assert 1 == vertex_2
    assert 2 == vertex_3

    edge_1 = graph_1.add_edge((0,1,1))
    egde_2 = graph_1.add_edge((1,2,3))
    assert edge_1[2] == 1
    assert egde_2[2] == 3