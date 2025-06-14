import sys
from operator import itemgetter

import networkx
from networkx.algorithms.community import asyn_lpa_communities
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity
import leidenalg
import igraph as ig

sys.path.append('/Users/abdulnaser/Desktop/Masterarbeit/metadatatransferlearning-main/meta_tl/')


def heaviest(G):
    u, v, w = max(G.edges(data="weight"), key=itemgetter(2))
    return u, v


def detect_communities_using_girvan_newman(graph):
    """
    Detect communities in the given graph using the Girvan-Newman algorithm.

    Parameters:
    graph (networkx.Graph): The graph on which to perform community detection.

    Returns:
    list: A list of communities, each community being a list of nodes.
    """
    # Apply the Girvan-Newman algorithm to find communities
    community_generator = girvan_newman(graph, most_valuable_edge=heaviest)

    # Get the first set of communities
    first_communities = next(community_generator)
    communities = [list(community) for community in first_communities]

    # Calculate and print the modularity
    modularity_value = modularity(graph, first_communities)

    # Print the number of communities and the modularity value
    print(f"Number of communities: {len(communities)}")
    print(f"Modularity: {modularity_value}")

    return communities


def detect_communities_using_label_propagation_clustering(graph):
    """
    Detect communities in the given graph using the Asynchronous Label Propagation algorithm
    and print the communities and modularity value.

    Parameters:
    graph (networkx.Graph): The graph on which to perform community detection.

    Returns:
    list: A list of communities, each community being a list of nodes.
    """
    # Detect communities using asynchronous label propagation
    communities = list(asyn_lpa_communities(graph, weight="weight"))

    # Calculate and print modularity
    modularity_value = modularity(graph, communities)

    # Print the number of communities and the modularity value
    print(f"Number of communities: {len(communities)}")
    print(f"Modularity: {modularity_value}")

    return communities


def detect_communities_using_louvain(graph):
    result = networkx.community.louvain_communities(graph)
    print(f"Number of communities: {len(result)}")
    return result


def detect_communities_using_leiden(graph: networkx.Graph):
    g = ig.Graph()
    print("|V|={} |E|={}".format(graph.number_of_nodes(), graph.number_of_edges()))
    for n in graph.nodes():
        g.add_vertices(str(n))
    for u, v in graph.edges():
        # if not has_node(g, str(u)):
        #     g.add_vertices(str(u))
        # if not has_node(g, str(v)):
        #     g.add_vertices(str(v))
        g.add_edge(str(u), str(v), weight=graph[u][v]['weight'], distance=graph[u][v]['distance'])
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, n_iterations=-1, weights='weight')
    nodes_in_cluster = set()
    communities = []
    for p in partition:
        cluster = set()
        for lp in p:
            cluster.add(int(lp))
            nodes_in_cluster.add(int(lp))
        communities.append(cluster)
    for n in graph.nodes():
        if n not in nodes_in_cluster:
            communities.append(set([n]))
    return communities


def has_node(graph, name):
    try:
        graph.vs.find(name=name)
    except:
        return False
    return True


def detect_communities(community_detection_algorithum, graph):
    if community_detection_algorithum == 'girvan_newman':
        communities = detect_communities_using_girvan_newman(graph)
        return communities

    elif community_detection_algorithum == 'label_propagation_clustering':
        communities = detect_communities_using_label_propagation_clustering(graph)
        return communities
    elif community_detection_algorithum == 'louvain':
        return detect_communities_using_louvain(graph)
    elif community_detection_algorithum == 'leiden':
        return detect_communities_using_leiden(graph)
