from networkx import Graph
import networkx as nx

def compute_diameter(graph:Graph, communities, selected_tasks:list, node_mapping:dict):
    node_diameter_dict = {}
    for community in communities:
        subgraph = graph.subgraph([node for node in community])
        diameter = nx.diameter(subgraph, weight='distance')
        for selected_task in selected_tasks:
            if node_mapping[str(selected_task)] in subgraph:
                node_diameter_dict[selected_task] = diameter
                break
    return node_diameter_dict
