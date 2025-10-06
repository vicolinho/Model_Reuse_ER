import os


import networkx
import networkx as nx
import numpy
from networkx import Graph

def evaluate(match_pairs:set[tuple[str,str]], weights: dict[tuple[str, str], float]):
    graph = Graph()
    a_set = set()
    b_set = set()
    num_all_edges = 0
    for t in match_pairs:
        w = weights[t]
        a_set.add(t[0])
        b_set.add(t[1])
        graph.add_edge(t[0], t[1], sim=w)
    strong_normal_edges = set()
    f_graph = filter_links(a_set, b_set, graph, types=["strong", "normal"])
    num_all_edges += graph.number_of_edges()
    for u, v in f_graph.edges():
        strong_normal_edges.add((u, v))
    est_tp = 0
    edge_labels = {}
    example_est = 0
    all_edges = graph.number_of_edges()
    tp_dict = {}
    for u, v in graph.edges():
        edge_labels[(u, v)] = round(graph[u][v]['sim'], 2)
        if (u, v) in strong_normal_edges:
            agg_sim_a = 0
            agg_sim_b = 0
            for u1, other_v in graph.edges(u):
                agg_sim_a += graph[u1][other_v]['sim']
            for other_u, v1 in graph.edges(v):
                agg_sim_b += graph[other_u][v1]['sim']
            p_u = graph[u][v]['sim'] / agg_sim_a
            p_v = graph[u][v]['sim'] / agg_sim_b
            est_tp += (p_u * p_v)
            tp_dict[(u, v)] = p_u * p_v
        else:
            tp_dict[(u, v)] = 0
    print(est_tp)
    print("#edges {}".format(num_all_edges))
    print("#edges {}".format((all_edges)))
    est_prec = est_tp / ((all_edges))
    return est_prec, tp_dict


def filter_links(a_set, b_set, graph: Graph, types=['normal', 'strong']):
    '''
    filters all weak links in a graph and build new clusters and graphs based on the connected components of the
    resulting graph
    :param types:
    :param cluster:
    :param graph:
    :return: updated_clusters, updated_graphs
    '''
    type_set = set(types)
    entity_resource_dict = dict()
    disjoint_resources = set()
    for e in graph.nodes():
        if e in a_set:
            disjoint_resources.add(0)
        elif e in b_set:
            disjoint_resources.add(1)
        if e not in entity_resource_dict:
            entity_resource_dict[e] = dict()
        resource_dict = entity_resource_dict[e]
        for edge in networkx.edges(graph, e):
            if e == edge[0]:
                other_node = edge[1]
            else:
                other_node = edge[0]
            if other_node in a_set:
                resource = 0
            elif other_node in b_set:
                resource = 1
            disjoint_resources.add(resource)
            if resource not in resource_dict:
                resource_dict[resource] = []
            entity_list: list = resource_dict[resource]
            sim = graph.get_edge_data(*edge)['sim']
            entity_list.append((other_node, sim))
        for res, ent_list in resource_dict.items():
            ent_list = sorted(ent_list, key=lambda v: v[1], reverse=True)
            resource_dict[res] = ent_list
    # normal links
    strong_normal_edges = set()
    normal_link_count = dict()
    strong_link_count = dict()
    for iri, resource_dict in entity_resource_dict.items():
        if iri in a_set:
            current_resource = 0
        elif iri in b_set:
            current_resource = 1
        for other_res, ent_list in resource_dict.items():
            max_sim = ent_list[0][1]
            found_strong = False
            for i in range(len(ent_list)):
                if max_sim != ent_list[i][1]:
                    break
                else:
                    other_ent_list = entity_resource_dict[ent_list[i][0]][current_resource]
                    other_max_sim = other_ent_list[0][1]
                    if other_max_sim == max_sim:
                        for k in range(len(other_ent_list)):
                            if other_max_sim != other_ent_list[k][1]:
                                break
                            else:
                                if iri == other_ent_list[k][0]:
                                    strong_edge = (iri, ent_list[i][0])
                                    rev_strong_edge = (ent_list[i][0], iri)
                                    strong_normal_edges.add(strong_edge)
                                    strong_normal_edges.add(rev_strong_edge)
                                    # found_strong = True
                                    #break
                    elif 'normal' in type_set:
                        strong_normal_edges.add((iri, ent_list[i][0]))
                        strong_normal_edges.add((ent_list[i][0], iri))
                    if found_strong:
                        break
    rem_edges = []
    for u, v in graph.edges():
        if (u, v) not in strong_normal_edges:
            rem_edges.append((u, v))
    rem_graph = graph.copy()
    rem_graph.remove_edges_from(rem_edges)
    return rem_graph


if __name__ == '__main__':
    evaluate()
