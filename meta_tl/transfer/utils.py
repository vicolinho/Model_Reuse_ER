import os

import networkx as nx
import numpy as np
import pandas as pd
from networkx import Graph
from sklearn.metrics.pairwise import euclidean_distances


# Calculates the Euclidean distance between a row and all rows in a DataFrame
def calculate_row_distances(row, df, columns_to_consider):
    distances = euclidean_distances([row[columns_to_consider]], df[columns_to_consider])
    return distances.flatten()


# Computes a weighted mean similarity score from a list of values
def compute_weighted_mean_similarity(values_list: list, weights=[], statistic_test='ks_test', model_score=1):
    if len(weights) == 0:
        weights = [1 for i in values_list]
    if type(values_list[0]) == float:
        values_transformed = [1 - x for x in values_list if type(x) == float]
    else:
        values_transformed = [1 - x[0] for x in values_list]
    # filtered_indices = [index for index, value in enumerate(values_transformed) if value == 3]
    filtered_indices = values_transformed
    filtered_values = [value for index, value in enumerate(values_transformed) if index not in filtered_indices]
    filtered_weights = [weight for index, weight in enumerate(weights) if index not in filtered_indices]
    # TODO check if it works
    threshold = (lambda x: x > 0.05) if statistic_test == 'ks_test' else (lambda x: x > 0.05)
    sim_vec = map(threshold, filtered_values)
    sim_vec = [a * b for a, b in zip(sim_vec, filtered_values)]
    #sim_vec = filtered_values
    if np.sum(filtered_weights) > 0:
        # return np.average(filtered_values, weights=filtered_weights)
        return np.average(sim_vec, weights=filtered_weights) * model_score
    else:
        return -1


def prepare_numpy_to_similarity_comparison_from_lp(linkage_problem: dict[(str, str):list]) -> np.ndarray:
    """
    Prepares the DataFrame for similarity comparison by removing the 'is_match' column,
    converting '/' to NaN, converting all columns to numeric, and filtering out columns
    with a high percentage of NaN values.

    Args:
        linkage_problem dict((str,str):list): The path to the CSV file.

    Returns:
        pd.DataFrame: The prepared DataFrame for similarity comparison.
    """
    # Set a threshold for the percentage of NaN values
    threshold_percentage = 70

    # Read the CSV file into a DataFrame
    sims = [l for l in linkage_problem.values()]
    # df = pd.DataFrame(np.asarray(sims), columns=[str(i) for i in range(len(sims[0]))])
    numpy_array = np.asarray(sims)
    numpy_array = numpy_array.astype(float)
    return numpy_array


# Prepares a DataFrame for prediction by cleaning and filtering columns
def prepare_dataframe_prediction(df, relevant_columns):
    df = df[relevant_columns]

    df.replace('/', np.nan, inplace=True)

    # Convert all columns to numerical data_io types
    df.iloc[:, 2:] = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

    return df


# Retrieves the number of rows in a similarity vector file
def get_sim_vec_file_length(path_to_sim_vector_folder, file_name):
    file_path = os.path.join(path_to_sim_vector_folder, file_name)
    file_df = pd.read_csv(file_path)
    return file_df.shape[0]


# Creates a graph from record linkage tasks
def create_graph(record_linkage_tasks, case, ratio_atomic_dis=0.5):
    # Create an empty undirected graph
    G = nx.Graph()
    # Extract unique values from the first and second columns
    first_column_nodes = set(record_linkage_tasks['first_task'].unique())
    second_column_nodes = set(record_linkage_tasks['second_task'].unique())
    if case == 3 or case == 1:
        filtered_tasks = record_linkage_tasks[record_linkage_tasks['similarity'] == 1]
    else:
        filtered_tasks = record_linkage_tasks[record_linkage_tasks['similarity'] >= ratio_atomic_dis]

    # Add nodes from the first column
    G.add_nodes_from(first_column_nodes)

    # Add nodes from the second column that are not in the first column
    unique_second_column_nodes = second_column_nodes - first_column_nodes
    G.add_nodes_from(unique_second_column_nodes)

    # Add edges from the 'first_file' and 'second_file' columns
    edges = filtered_tasks[['first_task', 'second_task', 'avg_similarity']].values.tolist()
    G.add_weighted_edges_from(edges)
    for e in edges:
        G[e[0]][e[1]]['distance'] = 1 - e[2]

    # Relabel nodes to consecutive numbers starting from 1
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G, mapping


def add_task(graph: Graph, record_linkage_tasks, mapping, case, ratio_atomic_dis=0.5):
    first_column_nodes = set(record_linkage_tasks['first_task'].unique())
    second_column_nodes = set(record_linkage_tasks['second_task'].unique())
    if case == 3 or case == 1:
        filtered_tasks = record_linkage_tasks[record_linkage_tasks['similarity'] == 1]
    else:
        filtered_tasks = record_linkage_tasks[record_linkage_tasks['similarity'] >= ratio_atomic_dis]
    for n in first_column_nodes.union(second_column_nodes):
        if str(n) not in mapping:
            node_id = len(mapping)
            graph.add_node(node_id)
            mapping[str(n)] = node_id
    edges = filtered_tasks[['first_task', 'second_task', 'avg_similarity']].values.tolist()
    for e in edges:
        e[0] = mapping[str(e[0])]
        e[1] = mapping[str(e[1])]
    graph.add_weighted_edges_from(edges)
    for e in graph.edges():
        graph[e[0]][e[1]]['distance'] = 1 - graph[e[0]][e[1]]['weight']
    return graph, mapping


def add_singleton_task(graph: Graph, singleton_task, mapping):
    if str(singleton_task) not in mapping:
        node_id = len(mapping)
        graph.add_node(node_id)
        mapping[str(singleton_task)] = node_id
    return graph, mapping


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='=', printEnd="\n"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
