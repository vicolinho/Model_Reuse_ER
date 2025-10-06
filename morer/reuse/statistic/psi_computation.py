'''
Code from
github repo population-stability-index ->https://github.com/mwburke/population-stability-index
'''
import argparse
import os

import numpy as np

import numpy as np
import pandas as pd

from morer.data_io import linkage_problem_io
from morer.reuse.incremental.util import split_linkage_problem_tasks
from morer.reuse.utils import prepare_numpy_to_similarity_comparison_from_lp

def calculate_psi(expected, actual, buckettype='bins', buckets=100, axis=0):
    bin_edges = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=buckets)

    # Calculate the proportions for each list in the same bins
    list_1_hist, _ = np.histogram(expected, bins=bin_edges)
    list_2_hist, _ = np.histogram(actual, bins=bin_edges)

    # Convert counts to proportions
    list_1_proportions = list_1_hist / len(expected)
    list_2_proportions = list_2_hist / len(actual)

    # PSI calculation: handle 0 counts by replacing with a small value to avoid division by zero
    epsilon = 1e-6
    psi_values = (list_1_proportions - list_2_proportions) * np.log(
        (list_1_proportions + epsilon) / (list_2_proportions + epsilon))

    # Total PSI
    psi_total = np.sum(psi_values)
    return psi_total

def calculate_psi_2(expected, actual, buckettype='bins', buckets=100, axis=0):
    '''Calculate the PSI (population stability index) across all variables

    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal

    Returns:
       psi_values: ndarray of psi values for each variable

    Author:
       Matthew Burke
       github.com/mwburke
       mwburke.github.io.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable

        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into

        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input

        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_fractions = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_fractions = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = sum(sub_psi(expected_fractions[i], actual_fractions[i]) for i in range(0, len(expected_fractions)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[1 - axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)


# if __name__ == '__main__':
#     # Example lists of values
#     parser = argparse.ArgumentParser(description='rl generation')
#     parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/dexter',
#                         help='linkage problem directory')
#     args = parser.parse_args()
#     MAIN_PATH = os.getcwd()
#
#     RECORD_LINKAGE_TASKS_PATH = os.path.join(MAIN_PATH, args.linkage_tasks_dir)
#     data_source_comp: dict[(str, str):[dict[(str, str):list]]] = linkage_problem_io.read_linkage_problems(
#         RECORD_LINKAGE_TASKS_PATH, deduplication=False)
#     linkage_problems = [(k[0], k[1], lp) for k, lp in data_source_comp.items()]
#     if 'dexter' in args.linkage_tasks_dir:
#         solved_problems, integrated_sources, unsolved_problems = split_linkage_problem_tasks(linkage_problems,
#                                                                                          split_ratio=0.5,
#                                                                                          is_shuffle=True)
#
#     linkage_problems_numpy_arrays = [prepare_numpy_to_similarity_comparison_from_lp(task[2]) for task in
#                                      solved_problems]
#     list_1 = np.asarray([0,0,0,0,0])  # Reference list (old data)
#     list_2 = np.asarray([1,1,1]) # Comparison list (new data)
#     # Define the number of bins (e.g., 10 bins)
#     n_bins = 10
#
#     # Compute the bin edges based on the combined data (to ensure bins match)
#     bin_edges = np.histogram_bin_edges(np.concatenate([list_1, list_2]), bins=n_bins)
#
#     # Calculate the proportions for each list in the same bins
#     list_1_hist, _ = np.histogram(list_1, bins=bin_edges)
#     list_2_hist, _ = np.histogram(list_2, bins=bin_edges)
#
#     # Convert counts to proportions
#     list_1_proportions = list_1_hist / len(list_1)
#     list_2_proportions = list_2_hist / len(list_2)
#
#     # PSI calculation: handle 0 counts by replacing with a small value to avoid division by zero
#     epsilon = 1e-6
#     psi_values = (list_1_proportions - list_2_proportions) * np.log(
#         (list_1_proportions + epsilon) / (list_2_proportions + epsilon))
#
#     # Total PSI
#     psi_total = np.sum(psi_values)
#
#     # Prepare results to display
#     psi_results = pd.DataFrame({
#         "Bin Range": [f"{bin_edges[i]:.2f} - {bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)],
#         "List 1 Proportion": list_1_proportions,
#         "List 2 Proportion": list_2_proportions,
#         "PSI": psi_values
#     })
#     psi = calculate_psi(list_1, list_2, buckets=10, axis=0)
#
#     print(psi)
#
#     print(psi_total)
#
