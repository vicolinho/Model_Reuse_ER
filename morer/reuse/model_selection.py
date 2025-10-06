import networkx as nx
import pandas as pd
import sys

from networkx import Graph

from morer.reuse.utils import *

def select_largest_lp_in_community(linkage_problem_dict: dict[(str, str):dict[(str, str):list]],
                                   community, node_mapping):
    """
    Select the largest file in each community and return the largest file and a list of other linkage tasks.

    Parameters:
    community (list): A list of nodes representing the community.
    node_mapping (dict): A dictionary mapping node labels to community nodes.

    Returns:
    tuple: A tuple containing the largest file and a list of other linkage tasks.
    """
    # Identify linkage tasks that belong to the community
    community_tasks = [task for task, node in node_mapping.items() if node in community]
    size_dict = {}
    for task in community_tasks:
        task_tuple = eval(task)
        task_count = len(linkage_problem_dict[task_tuple])
        if task_tuple[0] != task_tuple[1]:
            size_dict[task] = task_count
        else:
            size_dict[task] = 0
    # Create a list of other tasks in the community
    res = sorted(size_dict.items(), key=lambda x: x[1], reverse=True)
    selected_task = eval(res[0][0])
    # Create a list of other tasks in the community
    other_tasks = [(eval(task[0]), task[1]) for task in res]
    return selected_task, other_tasks


def select_task_with_largest_closeness_centrality(graph: Graph, community, reverse_node_mapping,
                                                  selection_type, linkage_problem_dict) -> (str, list):
    """
    Select the task with the largest centrality score and return the task with the largest centrality score.

    Parameters
    -------------
    graph : Graph
        A graph representing the graph.
    community : list
        A list of nodes representing the community.
    reverse_node_mapping : dict
        A dictionary mapping lp name to node labels.
    selection_type: str
        type for selecting the task for training data generation.

    :Returns:
    - selected lp - str
        name of the most relevant linkage problem
    - other_task - list
        order list of linkage problems regarding the relevance
    """
    # Identify linkage tasks that belong to the community
    subgraph = graph.subgraph([node for node in community])
    if selection_type == 'closeness_centrality':
        centrality = nx.closeness_centrality(subgraph, distance='distance')
    elif selection_type == 'betweenness_centrality':
        centrality = nx.betweenness_centrality(subgraph, weight='distance')
    elif selection_type == 'pageRank':
        centrality = nx.pagerank(subgraph, weight='weight')
    updated_dict = {}
    size_dict = {}
    for p in centrality.keys():
        task = eval(reverse_node_mapping[p])
        size = len(linkage_problem_dict[task])
        size_dict[p] = size
        if task[0] != task[1]:
            if selection_type == 'pageRank':
                updated_dict[p] = centrality[p]
            else:
                updated_dict[p] = centrality[p]
        else:
            updated_dict[p] = -1
    centrality = updated_dict
    total_size = sum(size_dict.values())
    for k, v in size_dict.items():
        if updated_dict[k] >= 0:
            updated_dict[k] = (updated_dict[k] + v / total_size)/2.0
    res = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    selected_task = eval(reverse_node_mapping[res[0][0]])
    # Create a list of other tasks in the community
    other_tasks = [(eval(reverse_node_mapping[task[0]]), task[1]) for task in res]
    return selected_task, other_tasks


def select_linkage_tasks_from_communities(linkage_problems: dict[(str, str):dict[(str, str):list]],
                                          linkage_tasks_communities: list[list], node_mapping,
                                          selection_strategy='largest',
                                          graph: Graph = None):
    """
    Loop over each community to select the largest file and related linkage tasks,
    then store the results in a dictionary.

    Parameters:
    linkage_tasks_communities (list): A list of communities, each community being a list of nodes.
    node_mapping (dict): A dictionary mapping node labels to community nodes.

    Returns:
    dict: A dictionary where keys are the largest file and values are lists of other linkage tasks.
    """
    selected_tasks_dict = {}
    if selection_strategy == 'largest':
        for community in linkage_tasks_communities:
            if len(community) > 1:
                largest_task, other_tasks = select_largest_lp_in_community(linkage_problems, community, node_mapping)
            else:
                reverse_node_mapping = {v: k for k, v in node_mapping.items()}
                largest_task = eval(reverse_node_mapping[list(community)[0]])
                other_tasks = [(largest_task, 0)]
            selected_tasks_dict[largest_task] = other_tasks
    elif 'centrality' in selection_strategy or 'pageRank' in selection_strategy:
        reverse_node_mapping = {v: k for k, v in node_mapping.items()}
        for community in linkage_tasks_communities:
            if len(community) > 1:
                largest_task, other_tasks = select_task_with_largest_closeness_centrality(graph, community,
                                                                                          reverse_node_mapping,
                                                                                          selection_strategy,
                                                                                          linkage_problems)
            else:
                largest_task = eval(reverse_node_mapping[list(community)[0]])
                other_tasks = [(largest_task, 0)]
            selected_tasks_dict[largest_task] = other_tasks
    return selected_tasks_dict
