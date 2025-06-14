import os
import sys
import warnings

import numpy
import xgboost as xgb
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics.pairwise import pairwise_kernels
from xgboost.sklearn import XGBClassifier

from meta_tl.transfer.statistic.psi_computation import calculate_psi

# Add the path to custom modules
from meta_tl.transfer.utils import compute_weighted_mean_similarity, \
    prepare_numpy_to_similarity_comparison_from_lp

# Suppress specific warning messages
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


def kl_divergence(p, q, num_bins=100):
    """Compute the KL divergence D(P || Q) using binned data.

    Parameters:
    p : numpy.ndarray
        First array (data points).
    q : numpy.ndarray
        Second array (data points).
    num_bins : int
        Number of bins to use for histogram.

    Returns:
    float
        KL divergence.
    """
    # Create histograms for both distributions
    hist_p, bin_edges = np.histogram(p, bins=num_bins, density=True)
    hist_q, _ = np.histogram(q, bins=bin_edges, density=True)

    # Normalize the histograms to ensure they sum to 1
    hist_p += 1e-10  # Avoid division by zero
    hist_q += 1e-10  # Avoid division by zero
    # Compute KL divergence
    kl_div = np.sum(hist_p * np.log(hist_p / hist_q))

    return kl_div


def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Parameters:
    X (ndarray): Array representing the first set of samples.
    Y (ndarray): Array representing the second set of samples.
    kernel (str or callable): Kernel function to use. Default is 'rbf' (Gaussian).
    gamma (float): Parameter for the RBF kernel. If None, it is inferred from data_io.

    Returns:
    float: The MMD value.
    """
    kernel = 'rbf' if kernel == 'squared_exp' else kernel
    K_XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    K_YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    K_XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)

    m, n = X.shape[0], Y.shape[0]

    mmd_squared = (np.sum(K_XX) / (m * (m - 1)) -
                   2 * np.sum(K_XY) / (m * n) +
                   np.sum(K_YY) / (n * (n - 1)))

    return np.sqrt(mmd_squared)


def mmd_permutation_test(X, Y, num_permutations=100, **kwargs):
    """
    Perform a permutation test to assess the significance of MMD between two sets of samples.

    Parameters:
    X (ndarray): Array representing the first set of samples.
    Y (ndarray): Array representing the second set of samples.
    num_permutations (int): Number of permutations to perform.
    **kwargs: Additional arguments to pass to the MMD function.

    Returns:
    float: The p-value of the permutation test.
    """
    mmd_observed = compute_mmd(X, Y, **kwargs)
    combined = np.vstack([X, Y])
    n_samples1 = X.shape[0]

    greater_extreme_count = 0
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        X_permuted, Y_permuted = combined[:n_samples1], combined[n_samples1:]
        mmd_permuted = compute_mmd(X_permuted, Y_permuted, **kwargs)
        if mmd_permuted >= mmd_observed:
            greater_extreme_count += 1

    return (greater_extreme_count + 1) / (num_permutations + 1)


# def calculate_psi(old_results, new_results):
#     """
#     Computes the Population Stability Index (PSI) to measure the shift in distributions between two datasets.
#
#     Parameters:
#     old_results (array-like): Observed values from the original dataset.
#     new_results (array-like): Observed values from the new dataset.
#
#     Returns:
#     float: The PSI value indicating the shift in distributions.
#     """
#
#     def psi(expected, actual):
#         return np.sum((actual - expected) * np.log(actual / expected))
#
#     old_expected = np.mean(old_results)
#     new_expected = np.mean(new_results)
#
#     return psi(old_expected, new_expected)


def compare_linkage_tasks(df_1: pd.DataFrame, df_2: pd.DataFrame, task_1_name, task_2_name, test_type,
                          stat_lists=None, multivariate=False):
    """
    Compares the distributions of two linkage tasks based on the specified test type.

    Parameters:
    lp1 (dict[(str, str):data frame]): dictionary of the first linkage problem with the record pairs as key and the similarities as data_frame.
    lp2 (dict[(str, str):data frame]): dictionary of the second linkage problem with the record pairs as key and the similarities as data_frame.
    task_1_name (str): Name of the first linkage task.
    task_2_name (str): Name of the second linkage task.
    test_type (str): The type of statistical test to perform ('ks_test', 'wasserstein_distance', 'calculate_psi', 'ML_based', or 'MMD').
    relevant_columns (list): List of relevant columns to compare.
    stat_lists (dict): Dictionary to store statistical test results for each column.
    multivariate (bool): Flag indicating if the test is multivariate.

    Returns:
    tuple: Names of the compared files and a list of resulting values from the statistical tests.
    """

    if multivariate:
        if test_type == 'ML_based':
            df_1['is_match'] = 0
            df_2['is_match'] = 1
            df_shuffled = pd.concat([df_1.sample(frac=1, random_state=42), df_2.sample(frac=1, random_state=42)],
                                    ignore_index=True)
            X, y = df_shuffled.drop(columns=['is_match']), df_shuffled['is_match']

            model = xgb.XGBClassifier()
            cv_score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()

            return task_1_name, task_2_name, cv_score

        elif test_type == 'MMD':
            df_1.apply(pd.to_numeric, errors='coerce').fillna(2)
            df_2.apply(pd.to_numeric, errors='coerce').fillna(2)
            mmd_value = mmd_permutation_test(df_1, df_2)
            return task_1_name, task_2_name, mmd_value

    else:
        intersection_columns = df_1.columns.intersection(df_2.columns)
        results = []
        for column_value in stat_lists:
            column = column_value
            if str(column) not in intersection_columns:
                stat_lists[column].append(-2)
            else:
                col1_is_nan = df_1.iloc[:, column].dropna().empty
                col2_is_nan = df_2.iloc[:, column].dropna().empty
                if col1_is_nan and col2_is_nan:
                    if test_type == 'ks_test':
                        stat_lists[column].append(0.06)
                        results.append(0.06)
                    elif test_type in ['wasserstein_distance', 'calculate_psi']:
                        stat_lists[column].append(0.9)
                        results.append(0.9)
                elif col1_is_nan or col2_is_nan:
                    if test_type == 'ks_test':
                        stat_lists[column].append(0.04)
                        results.append(0.04)
                    elif test_type in ['wasserstein_distance', 'calculate_psi']:
                        stat_lists[column].append(0.2)
                        results.append(0.2)
                else:
                    if test_type == 'ks_test':
                        ks_stat, p_value = ks_2samp(df_1.iloc[:, column].dropna(), df_2.iloc[:, column].dropna())
                        stat_lists[column].append(p_value)
                        results.append(p_value)
                    elif test_type == 'wasserstein_distance':
                        w_dist = wasserstein_distance(df_1.iloc[:, column].dropna(), df_2.iloc[:, column].dropna())
                        stat_lists[column].append(w_dist)
                        results.append(w_dist)
                    elif test_type == 'calculate_psi':
                        psi_value = calculate_psi(df_1.iloc[:, column].dropna(), df_2.iloc[:, column].dropna())
                        stat_lists[column].append(psi_value)
                        results.append(psi_value)
        return task_1_name, task_2_name, results


def compare_linkage_tasks_numpy(df_1: numpy.ndarray, df_2: numpy.ndarray, task_1_name, task_2_name, test_type,
                                stat_lists=None, multivariate=False):
    """
    Compares the distributions of two linkage tasks based on the specified test type.

    Parameters:
    lp1 (dict[(str, str):data frame]): dictionary of the first linkage problem with the record pairs as key and the similarities as data_frame.
    lp2 (dict[(str, str):data frame]): dictionary of the second linkage problem with the record pairs as key and the similarities as data_frame.
    task_1_name (str): Name of the first linkage task.
    task_2_name (str): Name of the second linkage task.
    test_type (str): The type of statistical test to perform ('ks_test', 'wasserstein_distance', 'calculate_psi', 'ML_based', or 'MMD').
    relevant_columns (list): List of relevant columns to compare.
    stat_lists (dict): Dictionary to store statistical test results for each column.
    multivariate (bool): Flag indicating if the test is multivariate.

    Returns:
    tuple: Names of the compared files and a list of resulting values from the statistical tests.
    """

    if multivariate:
        if test_type == 'ML_based':
            sample_1_idx = np.random.choice(df_1.shape[0], df_1.shape[0],
                                            replace=False)
            sample_2_idx = np.random.choice(df_2.shape[0], df_2.shape[0],
                                            replace=False)
            sample_1 = df_1[sample_1_idx, :]
            sample_2 = df_2[sample_2_idx, :]
            x = numpy.vstack((sample_1, sample_2))
            x_array = []
            for i in range(x.shape[0]):
                x_array.append(x[i])
            y = [1 for i in range(sample_1.shape[0])]
            y.extend([0 for i in range(sample_2.shape[0])])
            y = np.asarray(y)
            params = {
                'objective': 'binary:logistic',
                'max_depth': 4,
                'alpha': 10,
                'learning_rate': 1.0,
                'n_estimators': 100
            }
            indices = np.random.choice(x.shape[0], x.shape[0],
                                       replace=False)
            x = x[indices]
            y = y[indices]
            # instantiate the classifier
            model = XGBClassifier(**params)
            kfold = KFold(n_splits=5)
            cv_score = cross_val_score(model, x, y, cv=kfold, scoring='accuracy').mean()
            return task_1_name, task_2_name, cv_score
        elif test_type == 'MMD':
            mmd_value = mmd_permutation_test(df_1, df_2)
            return task_1_name, task_2_name, mmd_value

    else:
        intersection_columns = [i for i in range(min(df_1.shape[1], df_2.shape[1]))]
        results = []
        for column in stat_lists:
            if column not in intersection_columns:
                stat_lists[column].append(-2)
            else:
                col1_is_nan = False
                col2_is_nan = False
                # col1_is_nan = np.all(np.isnan(df_1[:, column]))
                # col2_is_nan = np.all(np.isnan(df_2[:, column]))
                if col1_is_nan and col2_is_nan:
                    if test_type == 'ks_test':
                        stat_lists[column].append(0.06)
                        results.append(0.06)
                    elif test_type in ['wasserstein_distance', 'calculate_psi']:
                        stat_lists[column].append(0.9)
                        results.append(0.9)
                elif col1_is_nan or col2_is_nan:
                    if test_type == 'ks_test':
                        stat_lists[column].append(0.04)
                        results.append(0.04)
                    elif test_type in ['wasserstein_distance', 'calculate_psi']:
                        stat_lists[column].append(0.2)
                        results.append(0.2)
                else:
                    col_array_1 = df_1[:, column]
                    col_array_2 = df_2[:, column]
                    # df_1_full = col_array_1[~np.isnan(col_array_1)]
                    # df_2_full = col_array_2[~np.isnan(col_array_2)]
                    if test_type == 'ks_test':
                        ks_stat, p_value = ks_2samp(col_array_1, col_array_2)
                        stat_lists[column].append((ks_stat, p_value))
                        results.append((ks_stat, p_value))
                    elif test_type == 'wasserstein_distance':
                        col_array_1[col_array_1 == -1] = 0
                        col_array_2[col_array_2 == -1] = 0
                        w_dist = wasserstein_distance(col_array_1, col_array_2)
                        stat_lists[column].append(w_dist)
                        results.append(w_dist)
                    elif test_type == 'calculate_psi':
                        col_array_1[col_array_1 == -1] = 0
                        col_array_2[col_array_2 == -1] = 0
                        max_value = calculate_psi(np.ones(col_array_1.shape), np.zeros(col_array_2.shape))
                        psi_value = calculate_psi(col_array_1, col_array_2)
                        scaled_psi_value = psi_value / max_value
                        stat_lists[column].append(scaled_psi_value)
                        results.append(scaled_psi_value)
                    elif test_type == 'kl_divergence':
                        col_array_1[col_array_1 == -1] = 0
                        col_array_2[col_array_2 == -1] = 0
                        kl_div = kl_divergence(col_array_1, col_array_2)
                        stat_lists[column].append(kl_div)
                        results.append(kl_div)
        return task_1_name, task_2_name, results


def evaluate_similarity(results, test_type, alpha=0.05):
    """
    Evaluates the similarity between two files based on the test results.

    Parameters:
    results (list or float): Resulting value(s) from the statistical tests.
    test_type (str): The type of statistical test performed.
    case (int): Case number to determine the evaluation logic (1 or 2).
    alpha (float): Significance level for the ks_test (default is 0.05).

    Returns:
    int: 1 if the files are similar, 0 otherwise.
    """
    if test_type == 'ML_based':
        return 0 if results > 0.80 else 1

    elif test_type == 'MMD':
        return 0 if results < 0.05 else 1

    elif isinstance(results, list):
        if test_type == 'ks_test':
            sim = 0 if any(value[1] < alpha for value in results) else 1
        else:  # 'wasserstein_distance' or 'calculate_psi'
            sim = 0 if any(value > 0.1 for value in results) else 1
        if sim == 0:
            threshold = (lambda x: x[1] > alpha) if test_type == 'ks_test' else (lambda x: x < 0.1)
            similar_count = sum(threshold(value) for value in results)
            #sim = 0.5 if similar_count >= len(results) // 2 else 0
            sim = round(similar_count / len(results), 3)
        return sim


def compute_similarity_test_numpy(test_type, linkage_problems: list[tuple[str, str, dict[(str, str):list]]],
                                  relevant_columns: list[int], multivariate=False, weights=[], is_save=False, path=''):
    """
    Computes the similarity between pairs of record linkage tasks using various statistical tests.

    Parameters:
    case (int): Determines the logic for evaluating similarity (1 or 2).
    test_type (str): The type of statistical test to perform.
    tasks_path (str): Path to the folder containing the record linkage tasks.
    relevant_columns (list): List of relevant columns to compare.
    multivariate (bool): Flag indicating if the test is multivariate.

    Returns:
    tuple: A DataFrame containing similar record linkage tasks and a general DataFrame with all comparisons.
    """
    stat_lists = {col: [] for col in relevant_columns} if not multivariate else None

    first_tasks, second_tasks, similarities, processed_pairs = [], [], [], []
    alpha = 0.05

    linkage_problems_numpy_arrays = [prepare_numpy_to_similarity_comparison_from_lp(task[2]) for task in
                                     linkage_problems]
    if len(weights) == 0:
        all_sims = np.vstack(linkage_problems_numpy_arrays)
        weights = np.std(all_sims, axis=0)
    processed = 0
    for index in tqdm(range(len(linkage_problems))):
        task_1 = linkage_problems[index]
        weight_vecs_1 = linkage_problems_numpy_arrays[index]
        for index_2 in range(index + 1, len(linkage_problems)):
            task_2 = linkage_problems[index_2]
            weight_vecs_2 = linkage_problems_numpy_arrays[index_2]
            if task_1[0] != task_2[0] and task_1[1] != task_2[1]:
                file1, file2, results = compare_linkage_tasks_numpy(
                    weight_vecs_1, weight_vecs_2, str((task_1[0], task_1[1])), str((task_2[0], task_2[1])),
                    test_type, stat_lists, multivariate)
                similarity = evaluate_similarity(results, test_type, alpha)
                first_tasks.append(file1)
                second_tasks.append(file2)
                similarities.append(similarity)
    return transform_to_statistic_result(first_tasks, second_tasks, stat_lists, similarities, multivariate,
                                         weights, is_save, path, test_type)


def transform_to_statistic_result(first_tasks, second_tasks, stat_lists, similarities, multivariate, weights,
                                  is_save, path, statistical_test, model_score=1):
    results_df = pd.DataFrame({
        'first_task': first_tasks,
        'second_task': second_tasks,
        **stat_lists,
        'similarity': similarities
    }) if not multivariate else pd.DataFrame({
        'first_task': first_tasks,
        'second_task': second_tasks,
        'similarity': similarities
    })
    #if case == 1:
    #    similar_tasks_df = results_df[results_df['similarity'] == 1]
    #elif case == 2 or case == 3:
    similar_tasks_df = results_df[results_df['similarity'] >= 0]
    if not multivariate:
        if len(results_df.index) > 0:
            results_df['avg_similarity'] = results_df.apply(
                lambda row: compute_weighted_mean_similarity(row[2:-1], weights, statistical_test, model_score), axis=1)
        else:
            print(results_df)
        if len(similar_tasks_df.index) > 0:
            similar_tasks_df['avg_similarity'] = similar_tasks_df.apply(
                lambda row: compute_weighted_mean_similarity(row[2:-1], weights, statistical_test, model_score), axis=1)
        else:
            print("empty")
    else:
        print(results_df['similarity'])

    # Output statistics
    num_similar_tasks = similar_tasks_df.shape[0]
    print(f"Number of tasks with similar distribution: {num_similar_tasks}")

    unique_first_tasks = similar_tasks_df['first_task'].unique()
    unique_second_tasks = similar_tasks_df['second_task'].unique()

    unique_tasks_count = len(set(unique_first_tasks) | set(unique_second_tasks))
    print(f"Total number of unique tasks with similar distribution: {unique_tasks_count}")
    if is_save:
        folder_stats = os.path.join(path, "statistical_tests_{}".format(statistical_test))
        print("save statistical tests at:" + folder_stats)
        if not os.path.exists(folder_stats):
            os.makedirs(folder_stats)
        similar_tasks_df.to_csv(os.path.join(folder_stats, "similar_tasks.csv"))
        results_df.to_csv(os.path.join(folder_stats, "results.csv"))
    return similar_tasks_df, results_df


def read_statistical_results(path: str, case: str) -> pd.DataFrame:
    folder_stats = os.path.join(path, "statistical_tests_{}".format(case))
    similar_tasks_df = pd.read_csv(os.path.join(folder_stats, "similar_tasks.csv"))
    results_df = pd.read_csv(os.path.join(folder_stats, "results.csv"))
    return similar_tasks_df, results_df
