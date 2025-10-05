import pandas as pd
from numpy import ndarray

from meta_tl.transfer.statistic import statistical_tests

def determine_similar_clusters(selected_solved_tasks: dict[(str, str):list[ndarray]], integrated_source: set,
                           unsolved_problems: dict[(str, str):ndarray], relevant_columns, multivariate=False,
                           test_type='ks_test', alpha=0.05, weights_dict={}, use_score=False, model_dict={}):

    for unsolved, lp_problem in unsolved_problems.items():
        cluster_results = []
        first_tasks, second_tasks, similarities, processed_pairs = [], [], [], []
        stat_lists = {col: [] for col in relevant_columns} if not multivariate else None
        for task, lps_problem_solved in selected_solved_tasks.items():
            for lp_problem_solved in lps_problem_solved:
                if unsolved[0].replace('_test', '') in integrated_source or unsolved[1].replace('_test', '') in integrated_source:
                    file1, file2, results = statistical_tests.compare_linkage_tasks_numpy(lp_problem, lp_problem_solved[1],
                                                                                          unsolved, lp_problem_solved[0], test_type,
                                                                                          stat_lists, multivariate)
                    first_tasks.append(file1)
                    second_tasks.append(file2)
                    similarity = statistical_tests.evaluate_similarity(results, test_type, alpha)
                    similarities.append(similarity)
            if use_score:
                model_score = model_dict[task][1]
            else:
                model_score = 1
            weights = weights_dict[task]
            similar_tasks_df, results_df = statistical_tests.transform_to_statistic_result(first_tasks, second_tasks,
                                                                                           stat_lists, similarities,
                                                                                           multivariate,
                                                                                           weights, False,
                                                                                           "",
                                                                                           statistical_test=test_type,
                                                                                           model_score=model_score)
            if len(similar_tasks_df.index) > 0:
                filtered_result = similar_tasks_df.groupby(['first_task', 'second_task'], as_index=False).agg({'similarity':'mean',
                                                                                              'avg_similarity': 'mean'})
                cluster_results.append(filtered_result)
        if len(cluster_results) > 0:
            filtered_result = pd.concat(cluster_results)
            if filtered_result.shape[0] > 0:
                selected_solved_problems_max = filtered_result[filtered_result['avg_similarity'] > 0]
                try:
                    if (selected_solved_problems_max['first_task'] == unsolved).all():
                        return None, unsolved, selected_solved_problems_max
                    elif (selected_solved_problems_max['second_task'] == unsolved).all():
                        return None, unsolved, selected_solved_problems_max
                except IndexError:
                    print(selected_solved_problems_max)
    print(unsolved_problems.keys())
    return None, None, None

def determine_clusters_with_sim(selected_solved_tasks: dict[(str, str):list[ndarray]], integrated_source: set,
                           unsolved_problems: dict[(str, str):ndarray], relevant_columns, multivariate=False,
                           test_type='ks_test', alpha=0.05, weights_dict={}, use_score=False, model_dict={}):

    for unsolved, lp_problem in unsolved_problems.items():
        cluster_results = []
        for task, lp_problem_solved in selected_solved_tasks.items():
            first_tasks, second_tasks, similarities, processed_pairs = [], [], [], []
            stat_lists = {col: [] for col in relevant_columns} if not multivariate else None
            # for lp_problem_solved in lp_problems_solved:
            if unsolved[0].replace('_test', '') in integrated_source or unsolved[1].replace('_test', '') in integrated_source:
                file1, file2, results = statistical_tests.compare_linkage_tasks_numpy(lp_problem, lp_problem_solved[0],
                                                                                      unsolved, task, test_type,
                                                                                      stat_lists, multivariate)
                first_tasks.append(file1)
                second_tasks.append(file2)
                similarity = statistical_tests.evaluate_similarity(results, test_type, alpha)
                similarities.append(similarity)
            if use_score:
                model_score = model_dict[task][1]
            else:
                model_score = 1
            weights = weights_dict[task]
            similar_tasks_df, results_df = statistical_tests.transform_to_statistic_result(first_tasks, second_tasks,
                                                                                           stat_lists, similarities,
                                                                                           multivariate,
                                                                                           weights, False,
                                                                                           "",
                                                                                           statistical_test=test_type,
                                                                                           model_score=model_score)
            if len(similar_tasks_df.index) > 0:
                filtered_result = similar_tasks_df.groupby(['first_task', 'second_task'], as_index=False).agg({'similarity':'mean',
                                                                                              'avg_similarity': 'mean'})
                cluster_results.append(filtered_result)
        if len(cluster_results) > 0:
            filtered_result = pd.concat(cluster_results)
            if filtered_result.shape[0] > 0:
                selected_solved_problems_max = filtered_result[filtered_result['avg_similarity'] > 0]
                try:
                    if (selected_solved_problems_max['first_task'] == unsolved).all():
                        return None, unsolved, selected_solved_problems_max
                    elif (selected_solved_problems_max['second_task'] == unsolved).all():
                        return None, unsolved, selected_solved_problems_max
                except IndexError:
                    print(selected_solved_problems_max)
    print(unsolved_problems.keys())
    return None, None, None


def determine_best_cluster(selected_solved_tasks: dict[(str, str):(ndarray, ndarray)], integrated_source: set,
                           unsolved_problems: dict[(str, str):ndarray], relevant_columns, multivariate=False,
                           test_type='ks_test', alpha=0.05, weights_dict={}, use_score=False, model_dict={}):

    for unsolved, lp_problem in unsolved_problems.items():
        cluster_results = []
        for task, lp_problem_solved in selected_solved_tasks.items():
            first_tasks, second_tasks, similarities, processed_pairs = [], [], [], []
            stat_lists = {col: [] for col in relevant_columns} if not multivariate else None
            # for lp_problem_solved in lp_problems_solved:
            if unsolved[0].replace('_test', '') in integrated_source or unsolved[1].replace('_test', '') in integrated_source:
                file1, file2, results = statistical_tests.compare_linkage_tasks_numpy(lp_problem, lp_problem_solved[0],
                                                                                      unsolved, task, test_type,
                                                                                      stat_lists, multivariate)
                first_tasks.append(file1)
                second_tasks.append(file2)
                similarity = statistical_tests.evaluate_similarity(results, test_type, alpha)
                similarities.append(similarity)
            if use_score:
                model_score = model_dict[task][1]
            else:
                model_score = 1
            weights = weights_dict[task]
            similar_tasks_df, results_df = statistical_tests.transform_to_statistic_result(first_tasks, second_tasks,
                                                                                           stat_lists, similarities,
                                                                                           multivariate,
                                                                                           weights, False,
                                                                                           "",
                                                                                           statistical_test=test_type,
                                                                                           model_score=model_score)
            if len(similar_tasks_df.index) > 0:
                filtered_result = similar_tasks_df.groupby(['first_task', 'second_task'], as_index=False).agg({'similarity':'mean',
                                                                                              'avg_similarity': 'mean'})
                cluster_results.append(filtered_result)
        if len(cluster_results) > 0:
            filtered_result = pd.concat(cluster_results)
            if filtered_result.shape[0] > 0:
                max_value = filtered_result['avg_similarity'].max()
                selected_solved_problems_max = filtered_result[filtered_result['avg_similarity'] == max_value]
                try:
                    if (selected_solved_problems_max['first_task'] == unsolved).all():
                        return selected_solved_problems_max['second_task'].values.tolist()[0], unsolved, selected_solved_problems_max
                    elif (selected_solved_problems_max['second_task'] == unsolved).all():
                        return selected_solved_problems_max['first_task'].values.tolist()[0], unsolved, selected_solved_problems_max
                except IndexError:
                    print(selected_solved_problems_max)
    print(unsolved_problems.keys())
    return None, None, None
