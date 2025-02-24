import os
import re
import math
import numpy as np
import pandas as pd
import networkx as nx
from networkx import Graph
from sklearn.metrics.pairwise import euclidean_distances


# Converts weight values from various units to pounds
def convert_weight_to_pounds(weight_str):
    if pd.isna(weight_str):
        return weight_str  # Return NaN as is
    if 'oz' in weight_str.lower():
        ounces = float(weight_str.split(' ')[0])  # Extract the number of ounces
        pounds = ounces / 16.0  # Convert ounces to pounds (1 pound = 16 ounces)
        return f'{pounds:.2f}'
    elif 'lbs' in weight_str.lower():
        return weight_str.split(' ')[0]  # Remove "lbs" suffix
    else:
        return 'Invalid'


# Consolidates weight values across multiple columns into a single column
def consolidate_weight_columns(row):
    for col in row.index:
        if not pd.isna(row[col]) and 'weight' in col:
            return row[col]
    return np.nan


# Cleans numeric columns by removing non-numeric characters and converting to numeric type
def clean_numeric_columns(df, columns):
    df[columns] = df[columns].replace(r'[^0-9.]', '', regex=True).apply(pd.to_numeric, errors='coerce')
    return df


# Consolidates dimension values across multiple columns into a single column
def consolidate_dimension_columns(row):
    for col in row.index:
        if not pd.isna(row[col]) and 'dimensions' in col:
            return row[col]
    return np.nan


# Converts various weight units to grams
def convert_weight_to_grams(weight_str):
    if pd.isna(weight_str):
        return weight_str  # Return NaN as is
    if 'oz' in weight_str.lower():
        ounces = float(weight_str.split(' ')[0])
        return ounces * 28.3495  # Convert ounces to grams
    elif 'lb' in weight_str.lower() or 'pounds' in weight_str.lower():
        pounds = float(weight_str.split(' ')[0])
        return pounds * 453.592  # Convert pounds to grams
    elif 'kg' in weight_str.lower():
        kilograms = float(weight_str.split(' ')[0])
        return kilograms * 1000  # Convert kilograms to grams
    else:
        return re.sub(r'gr', '', weight_str).replace('"', '').replace('\\', '')


# Extracts specific dimensions (width, height, depth) from a dimension string
def extract_dimension(dimensions_str, dimension, split_num):
    if pd.isna(dimensions_str):
        return dimensions_str
    splitt_array = dimensions_str.replace('″', '').strip().split('x')
    if 'cm' in dimensions_str:
        if dimension == 'width':
            width = splitt_array[split_num]
            return width
        if dimension == 'height':
            width = splitt_array[split_num]
            return width
        if dimension == 'depth':
            width = splitt_array[split_num]
            return width
    elif 'mm' in dimensions_str:
        if len(splitt_array) >= 3:
            if dimension == 'width':
                width = splitt_array[split_num] + ' mm'
                return width
            if dimension == 'height':
                width = splitt_array[split_num] + ' mm'
                return width
            if dimension == 'depth':
                width = splitt_array[split_num] + ' mm'
                return width
    else:
        return 'Invalid'


# Extracts resolution dimensions (width or height) from a resolution string
def extract_resolution(resolution_str, dimension):
    if pd.isna(resolution_str):
        return resolution_str
    dimensions = resolution_str.split('x')
    return dimensions[0] if dimension == 'x' else dimensions[1]


# Converts dimension values from millimeters to centimeters
def convert_mm_to_cm(dimension_str):
    dimension_str = str(dimension_str)
    if pd.isna(dimension_str):
        return dimension_str
    elif 'cm' in dimension_str:
        dimension_str_splitted = dimension_str.split(' ')
        return dimension_str_splitted[0]
    elif not any(char.isdigit() for char in dimension_str):
        return 0
    else:
        numeric_part = re.search(r'\d+', dimension_str)

        return float(numeric_part.group()) / 10.0


# Converts various dimension units to centimeters
def convert_dimension_to_cm(dim_str, unit=None):
    # Conversion factor from inches to centimeters
    cm_per_inch = 2.54

    if not pd.isna(dim_str):
        dim_str = dim_str.strip()

    if pd.isna(dim_str):
        return dim_str


    elif 'mm' in dim_str:
        dimension_str_splitted = dim_str.split(' ')
        return float(dimension_str_splitted[0]) / 10.0

    elif 'in' in dim_str or unit == 'inch':
        # Convert inches to centimeters
        cm = float(dim_str.split(' ')[0]) * cm_per_inch
        return cm
    elif 'cm' in dim_str:
        dimension_str_splitted = dim_str.split(' ')
        return dimension_str_splitted[0]

    else:
        processed_string = re.sub(r'[^\d.]', '', dim_str)
        return processed_string


# Extracts values for depth, width, and height from text using flexible patterns
def extract_dimensions_values(text):
    if pd.isna(text):
        return text

    values = {'d': None, 'w': None, 'h': None}
    matches = re.findall(r'(\d+\.\d+|\d+)[^\d]*(d|w|h)', text)

    for value, label in matches:
        values[label] = value

    return values


# Conditionally assigns a sensor type based on the input text
def assign_sensor_type(text):
    if pd.isna(text):
        return text
    if 'CMOS' in text:
        return 'CMOS'
    elif 'MOS' in text:
        return 'MOS'
    elif 'CCD' in text:
        return 'CCD'
    else:
        return math.nan


# Ensures the DataFrame has the specified columns, adding them if missing
def ensure_columns_exist(df, column_list):
    for col in column_list:
        if col not in df.columns:
            df[col] = "/"
    return df


# Deletes a file at the specified path
def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")


# Removes empty CSV files in the specified directory
def remove_empty_files(directory):
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        if df.empty:
            os.remove(file_path)


# Counts the number of files with a given extension in a directory
def count_files_in_directory(directory, extension='.csv'):
    return len([file for file in os.listdir(directory) if file.endswith(extension)])


# Counts the total number of records across all CSV files in a directory
def count_total_records(directory):
    total_records = 0
    files = [file for file in os.listdir(directory) if file.endswith('.csv')]
    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        total_records += df.shape[0]
    return total_records


# Deletes specified files or all files in a directory
def delete_files(directory, files_to_delete='ALL'):
    if files_to_delete == 'ALL':
        files = os.listdir(directory)
        if not files:
            print(f"No files to delete in {directory}")
            return
        for file in files:
            delete_file(os.path.join(directory, file))
        print(f"All files in {directory} have been deleted successfully.")
    else:
        delete_file(os.path.join(directory, files_to_delete))


# Calculates the Euclidean distance between a row and all rows in a DataFrame
def calculate_row_distances(row, df, columns_to_consider):
    distances = euclidean_distances([row[columns_to_consider]], df[columns_to_consider])
    return distances.flatten()


# Extracts the closest rows and calculates the match ratio for 'is_match' values
def calculate_match_ratio(row, df, columns_to_consider):
    distances = calculate_row_distances(row, df, columns_to_consider)
    closest_indices = np.argsort(distances)[1:10]
    closest_rows = df.iloc[closest_indices]
    match_count = (closest_rows['is_match'] == row['is_match']).sum()
    return match_count / len(closest_rows)


# Checks if an element is present in any list within a list of lists
def is_element_in_nested_list(nested_list, element):
    for inner_list in nested_list:
        if element in inner_list:
            return True, inner_list
    return False, []


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
        return np.average(sim_vec, weights=filtered_weights)*model_score
    else:
        return -1


# Prepares a DataFrame for similarity comparison by cleaning and filtering columns
def prepare_dataframe_to_similarity_comparison(file_path):
    """
    Prepares the DataFrame for similarity comparison by removing the 'is_match' column,
    converting '/' to NaN, converting all columns to numeric, and filtering out columns
    with a high percentage of NaN values.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The prepared DataFrame for similarity comparison.
    """
    # Set a threshold for the percentage of NaN values
    threshold_percentage = 70

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Check if 'is_match' column exists and remove it
    if 'is_match' in df.columns:
        df = df.drop(columns=['is_match'])

    # Replace "/" with NaN and convert all columns to numeric
    df.replace('/', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Filter columns where the percentage of NaN values is over the threshold
    filtered_columns = df.columns[df.isna().mean() < threshold_percentage / 100]
    df = df[filtered_columns]
    return df


# Prepares a DataFrame for similarity comparison by cleaning and filtering columns
def prepare_dataframe_to_similarity_comparison_from_lp(linkage_problem: dict[(str, str):list]):
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

    df = pd.DataFrame(np.asarray(sims), columns=[str(i) for i in range(len(sims[0]))])

    # Check if 'is_match' column exists and remove it
    if 'is_match' in df.columns:
        df = df.drop(columns=['is_match'])

    # Replace "/" with NaN and convert all columns to numeric
    df.replace('/', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Filter columns where the percentage of NaN values is over the threshold
    # filtered_columns = df.columns[df.isna().mean() < threshold_percentage / 100]
    # df = df[filtered_columns]
    return df


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
    mapping = {node: i + 1 for i, node in enumerate(G.nodes())}
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
        if n not in mapping:
            node_id = len(mapping) + 1
            graph.add_node(node_id)
            mapping[n] = node_id
    edges = filtered_tasks[['first_task', 'second_task', 'avg_similarity']].values.tolist()
    for e in edges:
        e[0] = mapping[e[0]]
        e[1] = mapping[e[1]]
    graph.add_weighted_edges_from(edges)
    for e in edges:
        graph[e[0]][e[1]]['distance'] = 1 - e[2]
    return graph


def add_singleton_task(graph: Graph, singleton_task, mapping):
    if singleton_task not in mapping:
        node_id = len(mapping) + 1
        graph.add_node(node_id)
        mapping[singleton_task] = node_id
    return graph


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
