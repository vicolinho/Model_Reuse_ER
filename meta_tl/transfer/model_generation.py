import time
import math
import operator
from collections import defaultdict
from operator import itemgetter

import numpy as np
import pandas as pd
import xgboost as xgb

from baseline.almser.ALMSER_EXP import ALMSER_EXP
from baseline.almser.almser_wrapper import transform_linkage_problems_to_df
from meta_tl.transfer.incremental.scoring import link_scoring
from record_linkage.classification.machine_learning import active_learning_solution
from record_linkage.classification.machine_learning.active_learning_solution import ActiveLearningBootstrap


def allocate_active_learning_budget(selected_tasks, linkage_problems: dict[(str, str):dict[(str, str):list]],
                                    tasks_info_df, min_budget_per_task, total_budget):
    """
    Allocate the active learning budget to both community (non-singleton) and singleton tasks.

    Parameters
    ----------
    selected_tasks : (dict)
        A dictionary where the keys are the most relevant LPS in the community (community leader),
    and the values are lists of CSV files in the community.
    linkage_problems: dictionary
        dictionary consisting of the similarity feature vectors for a LP
    tasks_info_df : pd.DataFrame
        DataFrame containing information about linkage tasks.
    min_budget_per_task : int
        The minimum budget to allocate per task.
    total_budget : int
        The total budget available for allocation.
    Returns
    ---------
    dict: A dictionary with task names as keys and their allocated budgets as values.
    """
    budget_allocation = {}
    community_sizes = {}

    # Calculate the total size of all files in each community (non-singleton tasks)
    non_singleton_tasks = set()
    singleton_tasks = set()
    for task_leader, community_tasks in selected_tasks.items():
        if len(community_tasks) > 1:
            # Calculate the size of the community leader directly
            leader_lp = linkage_problems[task_leader]
            leader_size = len(leader_lp)
            # Calculate the size of the rest of the community
            community_size = sum(
                len(linkage_problems[task[0]]) for task in community_tasks)
            # Combine the leader size with the rest of the community size
            community_sizes[task_leader] = community_size
            budget_allocation[task_leader] = community_size
            for task in community_tasks:
                non_singleton_tasks.add(str(task[0]))
        else:
            singleton_tasks.add(str(task_leader))
    number_of_community_tasks = len(budget_allocation)
    all_tasks = set(tasks_info_df['first_task'].unique()).union(tasks_info_df['second_task'].unique())
    print("non single tasks:{}".format(len(non_singleton_tasks)))
    print("unique tasks: {}".format(len(all_tasks) - len(non_singleton_tasks)))
    # Allocate budget for singleton tasks
    singleton_tasks = set()
    singleton_size = 0
    for task_string in all_tasks:
        task = eval(task_string)
        if task_string not in non_singleton_tasks:
            lp_problem = linkage_problems[task]
            singleton_tasks.add(task_string)
            singleton_size += len(lp_problem)
            budget_allocation[task] = len(lp_problem)
    total_min_budget_naive = min_budget_per_task * len(budget_allocation)
    total_min_budget_distribute = min_budget_per_task * number_of_community_tasks
    # Ensure the total minimum budget does not exceed the total available budget
    if total_min_budget_distribute > total_budget:
        raise ValueError("The total minimum budget of {} exceeds the total available budget of {}."
                         .format(total_min_budget_distribute, total_budget))

    # Calculate the remaining budget after allocating the minimum budget
    if total_min_budget_naive <= total_budget:
        remaining_budget = total_budget - total_min_budget_naive
    else:
        remaining_budget = total_budget - total_min_budget_distribute
    print("available budget to distribute: {}".format(remaining_budget))
    # Calculate the proportional allocations for the remaining budget based on combined community sizes
    total_size = sum(community_sizes.values()) + sum(
        budget_allocation[task] for task in budget_allocation if task not in community_sizes)
    proportional_allocations = {}

    print("total size of potential training data: {}".format(total_size))
    # Distribute budget for community tasks based on combined size

    overall_community_sizes = sum(community_sizes.values())
    remaining_budget_community = math.ceil(len(non_singleton_tasks) / len(all_tasks) * remaining_budget)
    #remaining_budget_community = math.ceil(overall_community_sizes / (overall_community_sizes+singleton_size) * remaining_budget)
    # assert total_size == overall_community_sizes+singleton_size
    remaining_budget_singleton = remaining_budget - remaining_budget_community
    print("community budget {} singleton budget: {}".format(remaining_budget_community, remaining_budget_singleton))

    for community_leader, total_community_size in community_sizes.items():
        proportional_allocations[community_leader] = (total_community_size / overall_community_sizes) * \
                                                     remaining_budget_community
    # fill all budgets for communities
    # 1. Calculate the remaining budget for single tasks
    # 2. Select the most dissimilar to assign extra budget.
    # 3. Calculate if the remaining budget is not empty and select a new one
    # adjust similarity if its higher than the remaining one
    # Distribute budget for singleton tasks
    if total_min_budget_naive <= total_budget:
        total_size_singleton = sum(
            budget_allocation[eval(task)] for task in singleton_tasks)
        for task in budget_allocation:
            if str(task) in singleton_tasks:  # Singleton tasks
                proportional_allocations[task] = (budget_allocation[
                                                      task] / total_size_singleton) * remaining_budget_singleton
    else:
        # avoid to merge the linkage tasks with the same data sources
        total_size_singleton = sum(
            budget_allocation[eval(task)] for task in singleton_tasks if eval(task)[0] != eval(task)[1])
        print("distribute remaining labels and merge singletons")
        print("total size singleton: {}".format(total_size_singleton))
        singleton_proportional_allocations, selected_tasks, budget_allocation = reassign_budget_for_singletons(
            remaining_budget_singleton,
            budget_allocation,
            tasks_info_df,
            selected_tasks,
            singleton_tasks, total_size_singleton)
        for task, alloc in singleton_proportional_allocations.items():
            assert task not in proportional_allocations
            proportional_allocations[task] = alloc
    # Combine the minimum budget with the proportional allocations and round up the final allocations
    final_allocations = {
        task: math.ceil(min_budget_per_task + proportional_allocations.get(task, 0)) for task in budget_allocation
    }
    print("final allocations:{}".format(sum(final_allocations.values())))
    return final_allocations, selected_tasks


def reassign_budget_for_singletons(remaining_budget, budget_allocation, task_info_df: pd.DataFrame,
                                   selected_tasks: dict, singleton_tasks, total_size):
    current_budget = remaining_budget
    proportional_allocations = {}
    total_size_recalc = 0
    not_same_data_source_tasks = set()
    for t in singleton_tasks:
        if eval(t)[0] != eval(t)[1]:
            total_size_recalc += budget_allocation[eval(t)]
            not_same_data_source_tasks.add(t)
    print("budget to distribute on single tasks:{}".format(remaining_budget))
    filtered_tasks_1 = task_info_df.loc[task_info_df['first_task'].isin(singleton_tasks)]
    filtered_tasks_2 = task_info_df.loc[task_info_df['second_task'].isin(singleton_tasks)]
    reverse_cluster = {}
    for k, v in selected_tasks.items():
        if len(v) > 1:
            for e in v:
                reverse_cluster[str(e[0])] = str(k)

    def assign_cluster(x, reverse_cluster):
        if x in reverse_cluster:
            return reverse_cluster[x]
        else:
            return -1

    filtered_tasks_1['cluster'] = filtered_tasks_1.apply(
        lambda x: assign_cluster(x['second_task'], reverse_cluster), axis=1)
    filtered_tasks_2['cluster'] = filtered_tasks_2.apply(
        lambda x: assign_cluster(x['first_task'], reverse_cluster), axis=1)
    no_sim_cluster_1 = filtered_tasks_1[filtered_tasks_1['cluster'] == -1]['first_task'].unique().tolist()
    no_sim_cluster_2 = filtered_tasks_2[filtered_tasks_2['cluster'] == -1]['second_task'].unique().tolist()
    sim_cluster_1 = filtered_tasks_1[filtered_tasks_1['cluster'] != -1]['first_task'].unique().tolist()
    sim_cluster_2 = filtered_tasks_2[filtered_tasks_2['cluster'] != -1]['second_task'].unique().tolist()
    singleton_with_atleast_no_cluster = set(no_sim_cluster_1).union(set(no_sim_cluster_2))
    singleton_with_atleast_one_cluster = set(sim_cluster_1).union(set(sim_cluster_2))
    task_with_no_cluster = singleton_with_atleast_no_cluster.difference(singleton_with_atleast_one_cluster)
    if len(task_with_no_cluster) > 0:
        for t in task_with_no_cluster:
            task = eval(t)
            task_budget = min((budget_allocation[task] / total_size) * remaining_budget, current_budget)
            proportional_allocations[task] = task_budget
            current_budget -= task_budget
    filtered_tasks_1.rename(columns={'first_task': 'single_task'}, inplace=True)
    filtered_tasks_2.rename(columns={'second_task': 'single_task'}, inplace=True)
    filtered_tasks = filtered_tasks_1.merge(filtered_tasks_2, how='outer')
    avg_singleton = filtered_tasks.groupby(["single_task", "cluster"], as_index=False).agg(
        {'avg_similarity': 'mean'})
    tuples = avg_singleton[["single_task", "cluster", "avg_similarity"]].values.tolist()
    avg_singleton = avg_singleton.loc[avg_singleton['single_task'].isin(not_same_data_source_tasks)]
    while current_budget > 0:
        min_sim = avg_singleton[avg_singleton.avg_similarity == avg_singleton.avg_similarity.min()]
        avg_singleton = avg_singleton[avg_singleton.avg_similarity != min_sim.avg_similarity.min()]
        min_list = min_sim['single_task'].tolist()
        for task_string in min_list:
            if task_string in singleton_tasks:
                task = eval(task_string)
                task_budget = min((budget_allocation[task] / total_size) * remaining_budget, current_budget)
                proportional_allocations[task] = task_budget
                current_budget -= task_budget
                if current_budget <= 0:
                    break
    node_cluster_dict = {}
    for t in tuples:
        if t[0] not in node_cluster_dict:
            node_cluster_dict[eval(t[0])] = []
        clusters = node_cluster_dict[eval(t[0])]
        clusters.append((t[1], t[2]))
    for task, clusters in node_cluster_dict.items():
        max_t = max(clusters, key=itemgetter(1))
        if task in selected_tasks and task not in proportional_allocations:
            assigned_cluster = selected_tasks[eval(max_t[0])]
            assigned_cluster.append((task, 0))
            selected_tasks[eval(max_t[0])] = assigned_cluster
            del selected_tasks[task]
            del budget_allocation[task]
    assert len(selected_tasks) == len(budget_allocation), str(len(budget_allocation)) + "-" + str(len(selected_tasks))
    return proportional_allocations, selected_tasks, budget_allocation


def generate_models(selected_tasks: dict[(str, str):list[(str, str)]],
                    linkage_problems: dict[(str, str):dict[(str, str):list]],
                    tasks_info_df, min_budget, iteration_budget, total_budget, gold_links, unsup_gold_links,
                    model_name='rf', active_learning_strategy='bootstrap'):
    """
    This method generates for each cluster C^i a model M_C^i. It examines the following steps:

    1. Budget allocation
        The total budget is distributed across the clusters. It distinguishes between singleton and clusters with more
        than 1 element. To guarantee sufficient training data, we allocate a minimum budget for each cluster. The remaining
        budget is proportional distributed regarding the size of the most relevant lp of the cluster.
        If the total budget is exceeded, we keep the most dissimilar singleton cluster regarding the other clusters as singleton
        and assign the remaining singletons to the most similar cluster.
    2. Active Learning
        For each cluster, we apply an active learning approach for the most relevant lp of the cluster.
        If we do not use the whole allocated budget, we repeat the active learning process with the 2nd relevant lp of
        the cluster and so on. We maintain all used lps for each cluster to use them for comparing with new lps.
    3. Model generation
        We build a random forest as classification model

    Parameters
    ----------
    selected_tasks  : dictionary
        A dictionary of clusters with the most relevant lp as key and the list of similar LPs as value.
    linkage_problems : dictionary
        dictionary of linkage problems with the data source pair as key and the dictionary of
        similarity features as value
    tasks_info_df : (pd.DataFrame)
        DataFrame containing the result of the statistical comparison
    min_budget : int
        The minimum budget to allocate per task.
    total_budget : int
        The total available budget.
    iteration_budget  : int
        budget for a batch.
    gold_links : set
        set of ground truth links as oracle
    model_name : str
        name of the model Default Random Forest
    ratio_sim_atomic_dis : filter for similar LPs




    : returns:
        - model_dict - model dictionary for each cluster.
        - selected_tasks - modified dictionary of clusters if we merged singleton to other clusters
        - training_data - dictionary with training data for each cluster
        - used_lps_for_training - dictionary with used LPs for each cluster


    """
    inter_scoring, intra_scoring = link_scoring.cluster_occurrence_scoring(linkage_problems, selected_tasks)
    model_dict = {}
    training_data_dict = {}
    selected_tasks_size = len(selected_tasks)
    # Allocate budgets to linkage tasks
    allocated_budgets, selected_tasks = allocate_active_learning_budget(selected_tasks, linkage_problems, tasks_info_df,
                                                                        min_budget,
                                                                        total_budget)
    print("allocated budget:{}".format(len(allocated_budgets)))
    print("clusters {}:".format(len(selected_tasks)))
    # Process each linkage task
    total_used_budget = 0

    for task, budget in allocated_budgets.items():
        lp_problem = linkage_problems[task]
        # Prepare the dataframe for prediction
        print(f"Processing task: {task}, initial shape: {len(lp_problem)}")
        used_budget = 0
        index = 0
        sorted_other_tasks = selected_tasks[task]
        lp_problem_extend = {}
        solved_problem_cluster = []
        record_pair_scoring = {}
        record_pair_scoring_intra = {}
        # TODO check candidate selection for cluster
        # while used_budget < budget and index < len(sorted_other_tasks):
        while index < len(sorted_other_tasks):
            lp_problem = linkage_problems[sorted_other_tasks[index][0]]
            lp_problem_extend.update(lp_problem)
            used_budget += len(lp_problem)
            if active_learning_strategy == 'QHC':
                solved_problem_cluster.append((sorted_other_tasks[index][0][0],
                                               sorted_other_tasks[index][0][1], lp_problem))
            for p in lp_problem.keys():
                record_pair_scoring[p] = (inter_scoring[p[0]] + inter_scoring[p[1]]) / 2.0
                task_dict = intra_scoring[task]
                record_pair_scoring_intra[p] = (task_dict[p[0]] + task_dict[p[1]]) / 2.0
            index += 1
        if index > 1:
            print("used more data sets: {}".format(index))
            # assert index == len(training_data_tasks)

        print("allocated budget {} for {} links".format(budget, len(lp_problem_extend)))
        if active_learning_strategy == 'bootstrap' or len(sorted_other_tasks) == 1:
            print("Bootstrap AL")
            active_learning = ActiveLearningBootstrap(budget, iteration_budget, k=100)
            current_train_vectors, current_train_class = active_learning.select_training_data(lp_problem_extend,
                                                                                              gold_links,
                                                                                              record_pair_scoring,
                                                                                              record_pair_scoring_intra)
        elif active_learning_strategy == 'almser':
            print("Almser AL")
            solved_problem_cluster = [(t[0][0], t[0][1], linkage_problems[t[0]]) for t in sorted_other_tasks]
            pairs_fv_train = transform_linkage_problems_to_df(solved_problem_cluster, '_', gold_links, unsup_gold_links)
            print("used training data:{}".format(len(pairs_fv_train.index)))
            pairs_fv_test = None
            # unique_source_pairs = set([str(t[0][0]) + '_' + str(t[0][1]) for tasks in selected_tasks.values() for t in tasks])
            unique_source_pairs = set(
                [str(t[0][0]) + '_' + str(t[0][1]) for t in sorted_other_tasks])
            print(unique_source_pairs)
            almser_exp = ALMSER_EXP(pairs_fv_train, pairs_fv_test, list(unique_source_pairs), budget, 'rf',
                                    'almser_gb', '_', None, bootstrap=True, details=False,
                                    batch_size=iteration_budget)
            almser_exp.run_AL(True)
            current_train_vectors, current_train_class = almser_exp.select_training_data()
        elif active_learning_strategy == 'QHC':
            pairs_fv_train = transform_linkage_problems_to_df(solved_problem_cluster, '_', gold_links, unsup_gold_links)
            print("used training data:{}".format(len(pairs_fv_train.index)))
            pairs_fv_test = None
            # unique_source_pairs = set([str(t[0][0]) + '_' + str(t[0][1]) for tasks in selected_tasks.values() for t in tasks])
            unique_source_pairs = set(
                [str(t[0][0]) + '_' + str(t[0][1]) for t in sorted_other_tasks])
            print(unique_source_pairs)
            almser_exp = ALMSER_EXP(pairs_fv_train, pairs_fv_test, list(unique_source_pairs), budget, 'rf',
                                    'disagreement', '_', None, bootstrap=True, details=False,
                                    batch_size=iteration_budget)
            almser_exp.run_AL(True)
            current_train_vectors, current_train_class = almser_exp.select_training_data()
        print("used budget {} vs calc budget {}".format(current_train_vectors.shape[0], budget))
        training_data_dict[task] = (current_train_vectors, current_train_class)
        cal_model, score = active_learning_solution.train_model(current_train_vectors, current_train_class, model_name,
                                                                False, True)
        model_dict[task] = (cal_model, score)
        total_used_budget += current_train_vectors.shape[0]
    print("used budget {}".format(total_used_budget))
    return model_dict, selected_tasks, training_data_dict


def select_new_training_data(allocated_budgets, tasks: dict[(str, str):list[(str, str, dict)]], linkage_problems,
                             iteration_budget, gold_links, unsup_gold_links,
                             model_name='rf', active_learning_strategy='bootstrap'):
    selected_tasks = defaultdict(list)
    for k, eps in tasks.items():
        for ep in eps:
            selected_tasks[k].append(((ep[0], ep[1]), 0))
    inter_scoring, intra_scoring = link_scoring.cluster_occurrence_scoring(linkage_problems, selected_tasks)
    total_used_budget = 0
    training_data_dict = {}
    for task, budget in allocated_budgets.items():
        # Prepare the dataframe for prediction
        used_budget = 0
        index = 0
        if budget > 0:
            lp_problem_extend = {}
            solved_problem_cluster = []
            record_pair_scoring = {}
            record_pair_scoring_intra = {}
            # TODO check candidate selection for cluster
            # while used_budget < budget and index < len(sorted_other_tasks):
            lp_list = tasks[task]
            for lp_problem in lp_list:
                lp_problem_extend.update(lp_problem[2])
                for p in lp_problem[2].keys():
                    record_pair_scoring[p] = (inter_scoring[p[0]] + inter_scoring[p[1]]) / 2.0
                    task_dict = intra_scoring[task]
                    record_pair_scoring_intra[p] = (task_dict[p[0]] + task_dict[p[1]]) / 2.0

            print("allocated budget {} for {} links".format(budget, len(lp_problem_extend)))
            if active_learning_strategy == 'bootstrap' or len(lp_list) == 1:
                print("Bootstrap AL")
                active_learning = ActiveLearningBootstrap(budget, iteration_budget, k=100)
                current_train_vectors, current_train_class = active_learning.select_training_data(lp_problem_extend,
                                                                                                  gold_links,
                                                                                                  record_pair_scoring,
                                                                                                  record_pair_scoring_intra)
            elif active_learning_strategy == 'almser':
                print("Almser AL")
                solved_problem_cluster = [(t[0], t[1], t[2]) for t in lp_list]
                pairs_fv_train = transform_linkage_problems_to_df(solved_problem_cluster, '_', gold_links,
                                                                  unsup_gold_links)
                print("used training data:{}".format(len(pairs_fv_train.index)))
                pairs_fv_test = None
                # unique_source_pairs = set([str(t[0][0]) + '_' + str(t[0][1]) for tasks in selected_tasks.values() for t in tasks])
                unique_source_pairs = set(
                    [str(t[0]) + '_' + str(t[1]) for t in lp_list])
                print(unique_source_pairs)
                almser_exp = ALMSER_EXP(pairs_fv_train, pairs_fv_test, list(unique_source_pairs), budget, 'rf',
                                        'almser_gb', '_', None, bootstrap=True, details=False,
                                        batch_size=iteration_budget)
                almser_exp.run_AL(True)
                current_train_vectors, current_train_class = almser_exp.select_training_data()
            elif active_learning_strategy == 'QHC':
                pairs_fv_train = transform_linkage_problems_to_df(solved_problem_cluster, '_', gold_links,
                                                                  unsup_gold_links)
                print("used training data:{}".format(len(pairs_fv_train.index)))
                pairs_fv_test = None
                # unique_source_pairs = set([str(t[0][0]) + '_' + str(t[0][1]) for tasks in selected_tasks.values() for t in tasks])
                unique_source_pairs = set(
                    [str(t[0]) + '_' + str(t[1]) for t in lp_list])
                print(unique_source_pairs)
                almser_exp = ALMSER_EXP(pairs_fv_train, pairs_fv_test, list(unique_source_pairs), budget, 'rf',
                                        'disagreement', '_', None, bootstrap=True, details=False,
                                        batch_size=iteration_budget)
                almser_exp.run_AL(True)
                current_train_vectors, current_train_class = almser_exp.select_training_data()
            print("used budget {} vs calc budget {}".format(current_train_vectors.shape[0], budget))
            training_data_dict[task] = (current_train_vectors, current_train_class)
            total_used_budget += current_train_vectors.shape[0]
    print("used budget {}".format(total_used_budget))
    return training_data_dict


def allocate_size(tasks: dict[(str, str), list[(str, str, dict)]], total_budget):
    allocated_budget = {}
    for task, eps in tasks.items():
        total_size = sum([len(task_t[2]) for task_t in eps])
        allocated_budget[task] = total_size
    all_links = sum(allocated_budget.values())
    for task in allocated_budget.keys():
        allocated_budget[task] = round(allocated_budget[task] / all_links * total_budget)
    return allocated_budget


def retrain_models(new_training_data: dict[(str, str):tuple], train_data_dict: dict[(str, str):tuple], model_name: str,
                   is_fine_tuned=True, is_calibrated=False):
    model_dict = {}
    for task, training_data in train_data_dict.items():
        if task in new_training_data:
            task_training_data = np.vstack((new_training_data[task][0], training_data[0]))
            task_labels = np.hstack((new_training_data[task][1], training_data[1]))
            labels = []
            for l in task_labels:
                if l:
                    labels.append(1)
                elif not l:
                    labels.append(0)
            task_labels = np.asarray(labels)
            assert task_training_data.shape[0] == task_labels.shape[0]
            try:
                model, score = active_learning_solution.train_model(task_training_data, task_labels, model_name,
                                                                    is_calibrated,
                                                                    is_fine_tuned)
            except ValueError as e:
                for i in range(task_labels.shape[0]):
                    if task_labels[i] or not task_labels[i]:
                        print(task_labels[i])
                print(e)
                exit(1)
            train_data_dict[task] = (task_training_data, task_labels)
            print("task {} trained with {} pairs instead of {} pairs".format(task, task_training_data.shape[0],
                                                                             training_data[0].shape[0]))
            model_dict[task] = (model, score)
    return model_dict, train_data_dict
