import argparse
import os
import sys
import time
import warnings
from collections import defaultdict
from statistics import mean

import networkx

sys.path.append(os.getcwd())
from morer.reuse import model_generation
import numpy as np
from numpy import std

from morer.data_io import linkage_problem_io
from morer.data_io.test_data import reader, wdc_reader, almser_linkage_reader
from morer.linkage_problem_analysis import util
from record_linkage.classification.machine_learning import constants, active_learning_solution
from record_linkage.evaluation import metrics
from morer.reuse.statistic import statistical_tests

from morer.reuse.incremental.cluster_selection import determine_best_cluster, determine_similar_clusters, \
    determine_clusters_with_sim
from morer.reuse.incremental.util import split_linkage_problem_tasks, \
    split_linkage_problem_tasks_on_training_data_pairs

# Ignore all warnings
warnings.filterwarnings("ignore")

MAIN_PATH = os.getcwd()

# Local application imports
from morer.reuse.utils import (
    add_task, create_graph, prepare_numpy_to_similarity_comparison_from_lp,
)
from morer.reuse.statistic.statistical_tests import compute_similarity_test_numpy
from graph_clustering import detect_communities
from model_selection import select_linkage_tasks_from_communities
from model_generation import generate_models
from evaluation import evaluate_predictions, evaluate_prediction_per_model

IS_BOOST = False

parser = argparse.ArgumentParser(description='rl generation')

parser.add_argument('--data_file', '-d', type=str, default='datasets/dexter/DS-C0/SW_0.3', help='data file')
parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/wdc_almser/train_pairs_fv.csv',
                    help='train pairs')
parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/wdc_almser/test_pairs_fv.csv',
                    help='test pairs')
# parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/music_almser/train_pairs_fv.csv',
#                     help='train pairs')
# parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/music_almser/test_pairs_fv.csv',
#                     help='test pairs')

parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/wdc_almser',
                    help='linkage problem directory')
# parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/dexter',
#                     help='linkage problem directory')
# parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/music_almser',
#                     help='linkage problem directory')
parser.add_argument('--statistical_test', '-s', type=str, default='ks_test',
                    choices=['ks_test', 'wasserstein_distance', 'calculate_psi', 'ML_based', 'kl_divergence'],
                    help='statistical test for comparing lps')
parser.add_argument('--ratio_sim_atomic_dis', '-rs', type=float, default=0,
                    help='amount of similar feature distributions so the lps are considered as similar')
parser.add_argument('--comm_detect', '-cd', type=str, default='leiden',
                    choices=['leiden', 'girvan_newman', 'label_propagation_clustering', 'louvain'],
                    help='communitiy detection algorithm')
parser.add_argument('--relevance_score', '-re', type=str, default='betweenness_centrality',
                    choices=['betweenness_centrality', 'largest', 'pageRank'],
                    help='relevance score for ordering the linkage problems in a cluster')
parser.add_argument('--model_generation', '-mg', type=str, default='bootstrap',
                    choices=['supervised', 'almser', 'bootstrap'],
                    help='active learning algorithm')
parser.add_argument('--budget_retrain', '-b_rt', type=int, default=250,
                    help='budget of retrain')
parser.add_argument('--min_budget', '-mb', type=int, default=50,
                    help='minimum budget for each cluster')
parser.add_argument('--total_budget', '-tb', type=int, default=1000,
                    help='total budget')
parser.add_argument('--unsolved_ratio', '-uns_ratio', type=float, default=0.25,
                    help='ratio between unsolved link features regarding pairs of a cluster')
parser.add_argument('--budget_unsolved', '-ub', type=int, default=200,
                    help='budget for unsolved linkage problems being not similar to any solved one')
parser.add_argument('--batch_size', '-b', type=int, default=5,
                    help='batch size')
parser.add_argument('--is_recluster', '-rec', type=bool, default=False,
                    help='recluster new ER problems')
parser.add_argument('--selection_strategy', '-sel', type=str, default='max', choices=['max', 'ensemble'],
                    help='selection_strategy')

args = parser.parse_args()
tps_list, fps_list, fn_list, p_list, r_list, f_list, run_t_list = [], [], [], [], [], [], []

# 'ks_test', 'wasserstein_distance', 'calculate_psi', 'ML_based', 'MMD'
STATISTICAL_TEST = args.statistical_test
FEATURE_CASE = 2
RATIO_SIM_ATOMIC_DIS = args.ratio_sim_atomic_dis
COMMUNITY_DETECTION_ALGORITHM = args.comm_detect  # or leiden, girvan_newman, 'label_propagation_clustering', louvain
MODEL_GENERATION_ALGORITHM = args.model_generation
ACTIVE_LEARNING_MIN_BUDGET = args.min_budget
ACTIVE_LEARNING_ITERATION_BUDGET = args.batch_size
ACTIVE_LEARNING_TOTAL_BUDGET = args.total_budget
BUDGET_RETRAIN = args.budget_retrain
IS_RECLUSTER = args.is_recluster
LP_SELECTION_STRATEGY = args.selection_strategy
UNSOLVED_RATIO = args.unsolved_ratio
ACTIVE_LEARNING_MIN_UNSOLVED = args.budget_unsolved

ACTIVE_LEARNING_ITERATION_BUDGET_UNSOLVED = args.batch_size
SELECTION_STRATEGY = args.relevance_score
multivariate = STATISTICAL_TEST in ['ML_based', 'MMD']

for i in range(3):
    unsupervised_gold_links = set()
    if 'dexter' in args.linkage_tasks_dir:
        file_name = os.path.join(MAIN_PATH, args.data_file)
        entities, _, _ = reader.read_data(file_name)
        gold_clusters = reader.generate_gold_clusters(entities)
        gold_links = metrics.generate_links(gold_clusters)
        data_sources_dict, data_sources_headers = reader.transform_to_data_sources(entities)
    elif 'wdc_computer' in args.linkage_tasks_dir:
        train_tp_links, train_tn_links, test_tp_links, test_tn_links = wdc_reader.read_wdc_links(args.train_pairs,
                                                                                                 args.test_pairs)
        gold_links = set()
        gold_links.update(train_tp_links)
        gold_links.update(test_tp_links)
    elif 'wdc_almser' in args.linkage_tasks_dir or 'music_almser' in args.linkage_tasks_dir:
        gold_links = set()
        train_tp_links, train_tn_links, test_tp_links, test_tn_links, unsup_train_tp_links, unsup_train_tn_links = (
            almser_linkage_reader.read_wdc_links(args.train_pairs, args.test_pairs))
        gold_links.update(train_tp_links)
        gold_links.update(test_tp_links)
        unsupervised_gold_links.update(unsup_train_tp_links)
        # print("tps overall {}".format(len(gold_links)))
    unsupervised_gold_links = set()
    RECORD_LINKAGE_TASKS_PATH = os.path.join(MAIN_PATH, args.linkage_tasks_dir)
    ML_MODEL = constants.RF
    data_source_comp: dict[(str, str):[dict[(str, str):list]]] = linkage_problem_io.read_linkage_problems(
        RECORD_LINKAGE_TASKS_PATH, deduplication=False)
    all_pairs = set([p for lp in data_source_comp.values() for p in lp.keys()])
    # ===================================================
    # Step 1: Prepare Record Linkage Tasks
    # ===================================================

    reduced_comp = linkage_problem_io.remove_empty_problems(data_source_comp)

    # Count and # print the total number of record pairs across all tasks
    total_record_pairs = util.count_total_number_of_links(reduced_comp)

    # ===================================================
    # Step 2: Perform Linkage Tasks Distribution Test
    # ===================================================
    start_overall_time = time.time()

    linkage_problems = [(k[0], k[1], lp) for k, lp in data_source_comp.items()]
    relevant_columns = [col_index for col_index in range(len(list(linkage_problems[0][2].values())[0]))]
    # split linkage problems to solved problems and to unsolved
    if 'dexter' in args.linkage_tasks_dir:
        solved_problems, integrated_sources, unsolved_problems = split_linkage_problem_tasks(linkage_problems,
                                                                                             split_ratio=0.5,
                                                                                             is_shuffle=True)
    elif 'wdc_computer' in args.linkage_tasks_dir:
        train_check = set(train_tp_links)
        train_check.update(train_tn_links)
        test_check = set(test_tp_links)
        test_check.update(test_tn_links)
        solved_problems, integrated_sources, unsolved_problems, data_source_comp = (
            split_linkage_problem_tasks_on_training_data_pairs
            (data_source_comp, train_check, test_check))
        removed_pairs = set(data_source_comp.keys()).difference(
            set([(t[0], t[1]) for t in solved_problems]).union([(t[0], t[1]) for t in unsolved_problems]))
        for t in removed_pairs:
            del data_source_comp[t]
        assert len(solved_problems) + len(unsolved_problems) == len(data_source_comp), (str(len(data_source_comp)) +
                                                                                        "  " + str(
                    len(solved_problems) + len(unsolved_problems)))
    elif 'wdc_almser' in args.linkage_tasks_dir or 'music_almser' in args.linkage_tasks_dir:
        solved_problems = []
        unsolved_problems = []
        integrated_sources = set()
        tps_check = 0
        for lp, sims in data_source_comp.items():
            if 'train' in lp[0]:
                solved_problems.append((lp[0], lp[1], sims))
                for p in sims.keys():
                    if p in gold_links:
                        tps_check += 1
                integrated_sources.add(lp[0].replace('_train', ''))
                integrated_sources.add(lp[1].replace('_train', ''))
            if 'test' in lp[0]:
                unsolved_problems.append((lp[0], lp[1], sims))

    weights = {}
    linkage_problems_numpy_arrays = [prepare_numpy_to_similarity_comparison_from_lp(task[2]) for task in
                                     solved_problems]
    all_sims = np.vstack(linkage_problems_numpy_arrays)
    weights = np.std(all_sims, axis=0)
    weights[weights == 0] = 0.05

    if not os.path.exists(
            os.path.join(RECORD_LINKAGE_TASKS_PATH, "incremental/statistical_tests_{}".format(STATISTICAL_TEST))):
        linkage_tasks_similarity_df, linkage_tasks_general_df = compute_similarity_test_numpy(
            STATISTICAL_TEST, solved_problems, relevant_columns, multivariate=multivariate, weights=weights,
            is_save=False,
            path=os.path.join(RECORD_LINKAGE_TASKS_PATH, 'incremental')
        )
    else:
        linkage_tasks_similarity_df, linkage_tasks_general_df = statistical_tests \
            .read_statistical_results(path=os.path.join(RECORD_LINKAGE_TASKS_PATH, 'incremental'),
                                      case=STATISTICAL_TEST)

    # ===================================================
    # Step 3: Perform Graph Clustering on Linkage Tasks
    # ===================================================

    # Create a graph of linkage tasks based on their similarity
    # print(linkage_tasks_general_df)
    graph, task_mapping = create_graph(linkage_tasks_general_df, case=FEATURE_CASE,
                                       ratio_atomic_dis=RATIO_SIM_ATOMIC_DIS)
    reverse_new_mapping = {v: k for k, v in task_mapping.items()}
    # Detect communities within the graph using the selected algorithm
    linkage_task_communities = detect_communities(
        COMMUNITY_DETECTION_ALGORITHM, graph
    )
    lp_quartile_map = {}
    for c in linkage_task_communities:
        sub_graph = networkx.subgraph(graph, c)
        sims = [sub_graph[u][v]['distance'] for u, v, d in sub_graph.edges(data=True)]
        quartile = np.quantile(np.asarray(sims), 0.75)
        for n in c:
            lp_quartile_map[eval(reverse_new_mapping[n])] = quartile
    # ===================================================
    # Step 4: Select Linkage Tasks from Each Community
    # ===================================================

    # Select the largest task as representative as key and a order list of tasks based on a certain relevance criterion
    selected_tasks = select_linkage_tasks_from_communities(
        data_source_comp, linkage_task_communities, task_mapping,
        selection_strategy=SELECTION_STRATEGY, graph=graph
    )
    # ===================================================
    # Step 5: Apply Active Learning to Label Selected Tasks
    # ===================================================

    # Record the start time for active learning
    start_time = time.time()

    # Apply active learning to label the selected tasks (uncomment when ready)
    lps_covered_by_training_data = set()
    cal_model_dict, selected_tasks, train_data_dict = generate_models(
        selected_tasks, data_source_comp, linkage_tasks_general_df,
        min_budget=ACTIVE_LEARNING_MIN_BUDGET, iteration_budget=ACTIVE_LEARNING_ITERATION_BUDGET,
        total_budget=ACTIVE_LEARNING_TOTAL_BUDGET, gold_links=gold_links, unsup_gold_links=unsupervised_gold_links,
        model_name=ML_MODEL, active_learning_strategy=MODEL_GENERATION_ALGORITHM)

    # print("Finished preparation for solved linkage problems")
    # print("Start to solve new linkage problems")
    weight_dict = {c_id: weights for c_id in cal_model_dict.keys()}
    used_budgets_unsolved = 0
    result_dictionary = {}
    results_per_model = {}
    selected_numpy_dict = {}
    # ===================================================
    # Step 6.0: Prepare solved and unsolved tasks by initializing numpy arrays
    # Utilize selected training data feature vectors to determine similarity between cluster and unsolved linkage problem
    # ===================================================
    if IS_RECLUSTER:
        for selected_task, lps in selected_tasks.items():
            cluster_lps = []
            for other_task, centrality in lps:
                lp = data_source_comp[other_task]
                lp_numpy = statistical_tests.prepare_numpy_to_similarity_comparison_from_lp(lp)
                lps_covered_by_training_data.add(other_task)
                cluster_lps.append((other_task, lp_numpy))
            selected_numpy_dict[selected_task] = cluster_lps
    else:
        selected_numpy_dict = train_data_dict

    unsolved_numpy_dict = {}
    debug_labels = {}
    for unsolved_task in unsolved_problems:
        lp = unsolved_task[2]
        lp_numpy = statistical_tests.prepare_numpy_to_similarity_comparison_from_lp(lp)
        lp_labels = [tuple(sorted(k)) in gold_links for k in lp.keys()]
        unsolved_numpy_dict[(unsolved_task[0], unsolved_task[1])] = lp_numpy
        debug_labels[(unsolved_task[0], unsolved_task[1])] = np.asarray(lp_labels)

    # ===================================================
    # Step 6: Apply model search for unsolved problems
    # ===================================================
    new_lps = defaultdict(list)
    total_tasks = len(unsolved_numpy_dict)
    solved_tasks = 0
    labeled_data = 0
    while len(unsolved_numpy_dict) != 0:
        # select most similar task
        if not IS_RECLUSTER and LP_SELECTION_STRATEGY == 'max':
            # print("only one selected task -> is recluster: {} selection: {}".format(IS_RECLUSTER, LP_SELECTION_STRATEGY))
            selected_task, unsolved_problem, linkage_task_sims = determine_best_cluster(selected_numpy_dict,
                                                                                        integrated_sources,
                                                                                        unsolved_numpy_dict,
                                                                                        relevant_columns,
                                                                                        multivariate, STATISTICAL_TEST,
                                                                                        weights_dict=weight_dict,
                                                                                        use_score=False, model_dict=
                                                                                        cal_model_dict)
            selected_tasks_list = {selected_task: 1}
        elif not IS_RECLUSTER and LP_SELECTION_STRATEGY == 'ensemble':
            # print("multiple selected tasks -> is recluster: {} selection: {}".format(IS_RECLUSTER, LP_SELECTION_STRATEGY))
            selected_task, unsolved_problem, linkage_task_sims = determine_clusters_with_sim(selected_numpy_dict,
                                                                                             integrated_sources,
                                                                                             unsolved_numpy_dict,
                                                                                             relevant_columns,
                                                                                             multivariate,
                                                                                             STATISTICAL_TEST,
                                                                                             weights_dict=weight_dict,
                                                                                             use_score=False,
                                                                                             model_dict=
                                                                                             cal_model_dict)
            selected_tasks_sims = linkage_task_sims[['first_task', 'second_task', 'avg_similarity']].values.tolist()
            selected_tasks_with_weights = {}
            total_sum = sum(tup[2] for tup in selected_tasks_sims)
            for pair in selected_tasks_sims:
                if pair[0] == unsolved_problem:
                    selected_tasks_with_weights[pair[1]] = pair[2] / total_sum
                else:
                    selected_tasks_with_weights[pair[0]] = pair[2] / total_sum
        elif IS_RECLUSTER:
            print("recluster graph -> is recluster: {} selection: {}".format(IS_RECLUSTER, LP_SELECTION_STRATEGY))
            selected_task, unsolved_problem, linkage_task_sims = determine_similar_clusters(selected_numpy_dict,
                                                                                            integrated_sources,
                                                                                            unsolved_numpy_dict,
                                                                                            relevant_columns,
                                                                                            multivariate,
                                                                                            STATISTICAL_TEST,
                                                                                            weights_dict=weight_dict,
                                                                                            use_score=False,
                                                                                            model_dict=
                                                                                            cal_model_dict)

            # quantile_new = np.quantile(np.asarray(sims_new), 0.25)
            # print("new quantile:{}".format(quantile_new))
        if IS_RECLUSTER:
            graph, task_mapping = add_task(graph, linkage_task_sims, task_mapping, FEATURE_CASE,
                                           ratio_atomic_dis=RATIO_SIM_ATOMIC_DIS)
            reverse_new_mapping = {v: k for k, v in task_mapping.items()}
            new_cluster = detect_communities(
                COMMUNITY_DETECTION_ALGORITHM, graph)
            new_cluster_sets = [set([eval(reverse_new_mapping[n]) for n in cluster]) for cluster in new_cluster]
            rev_selected_tasks = {v[0]: k for k, ots in selected_tasks.items() for v in ots}

            ## print(rev_selected_tasks)
            selected_tasks_list = {v: 0 for v in rev_selected_tasks.values()}
            unsolved_cluster = None
            for other_task, head_task in rev_selected_tasks.items():
                for c in new_cluster_sets:
                    if other_task in c and unsolved_problem in c:
                        selected_task = head_task
                        selected_tasks_list[head_task] = selected_tasks_list[head_task] + 1
                        break
                    elif unsolved_problem in c:
                        unsolved_cluster = c
        if unsolved_problem is not None:
            integrated_sources.add(unsolved_problem[0].replace('_test', ''))
            integrated_sources.add(unsolved_problem[1].replace('_test', ''))
        # print(len(unsolved_numpy_dict))
        # print("selection: {} for unsolved problem {}".format(selected_tasks_list, unsolved_problem))
        # apply model on unsolved model
        if LP_SELECTION_STRATEGY == 'max' or IS_RECLUSTER:
            if len(selected_tasks_list) > 0:
                selected_task = sorted(selected_tasks_list, key=selected_tasks_list.get, reverse=True)[0]
                selected_cluster = None
                lp_problem = data_source_comp[unsolved_problem]
                if IS_RECLUSTER:
                    for c in new_cluster:
                        if task_mapping[str(unsolved_problem)] in c:
                            selected_cluster = c
                            break
                    new_data = new_lps[selected_task]
                    new_data.append((unsolved_problem[0], unsolved_problem[1], lp_problem))
                    solved_lps_count = sum(features[1].shape[0] for features in selected_numpy_dict[selected_task])
                    # print(solved_lps_count)
                    new_lp_quartile_map = {}
                    for c in new_cluster:
                        sub_graph = networkx.subgraph(graph, c)
                        sims = [sub_graph[u][v]['distance'] for u, v, d in sub_graph.edges(data=True)]
                        quartile = np.quantile(np.asarray(sims), 0.75)
                        for n in c:
                            new_lp_quartile_map[eval(reverse_new_mapping[n])] = quartile
                    old_quartile_avg = 0
                    new_quartile_avg = 0
                    counter = 0
                    for n in selected_cluster:
                        if eval(reverse_new_mapping[n]) in lp_quartile_map:
                            old_quartile_avg += (1 - lp_quartile_map[eval(reverse_new_mapping[n])])
                            new_quartile_avg += (1 - new_lp_quartile_map[eval(reverse_new_mapping[n])])
                            counter += 1
                    old_quartile_avg /= counter
                    new_quartile_avg /= counter
                    print("old sim quartile {} - new sim quartile {}".format(old_quartile_avg, new_quartile_avg))
                    lp_quartile_map = new_lp_quartile_map
                    unsolved_lps_count = 0
                    for features in new_lps[selected_task]:
                        unsolved_lps_count += (len(features[2]))
                    covered_ratio = unsolved_lps_count / (unsolved_lps_count + solved_lps_count)
                    print("{} - {} ratio of links from solved lps: {}".format(selected_task, unsolved_problem,
                                                                              covered_ratio))
                    print("covered ratio: {}".format(UNSOLVED_RATIO != -1))
                    if covered_ratio > UNSOLVED_RATIO != -1:
                        print("Retrain")
                        old_train_features, old_train_labels = train_data_dict[selected_task]
                        ratio_training_data = old_train_features.shape[0] / ACTIVE_LEARNING_TOTAL_BUDGET
                        allocated_budget = {selected_task: int(ACTIVE_LEARNING_TOTAL_BUDGET * covered_ratio *
                                                               ratio_training_data)}
                        used_budgets_unsolved += allocated_budget[selected_task]
                        new_tasks = {selected_task: new_lps[selected_task]}
                        new_training_data = model_generation.select_new_training_data(allocated_budget, new_tasks,
                                                                                      data_source_comp,
                                                                                      ACTIVE_LEARNING_ITERATION_BUDGET,
                                                                                      gold_links,
                                                                                      unsupervised_gold_links, ML_MODEL,
                                                                                      MODEL_GENERATION_ALGORITHM)
                        for t in new_training_data.values():
                            labeled_data += t[0].shape[0]
                        print("old model data: {}".format(train_data_dict[selected_task][1].shape))
                        new_model_dict, train_data_dict = (
                            model_generation.retrain_models(new_training_data, train_data_dict, ML_MODEL,
                                                            is_fine_tuned=True,
                                                            is_calibrated=False))
                        print("new model data: {}".format(train_data_dict[selected_task][1].shape))
                        cal_model_dict.update(new_model_dict)
                        lp_list = selected_numpy_dict[selected_task]
                        # print(unsolved_numpy_dict.keys())
                        for uns_lp in new_lps[selected_task]:
                            uns_lp_numpy = statistical_tests.prepare_numpy_to_similarity_comparison_from_lp(
                                data_source_comp[(uns_lp[0], uns_lp[1])])
                            lp_list.append(((uns_lp[0], uns_lp[1]), uns_lp_numpy))
                            selected_tasks[selected_task].append(((uns_lp[0], uns_lp[1]), 0))
                        selected_numpy_dict[selected_task] = lp_list
                        del new_lps[selected_task]
                model = cal_model_dict[selected_task][0]
                lp_problem = data_source_comp[unsolved_problem]
                class_match_set, class_non_match_set, pair_confidences = active_learning_solution \
                    .classify(lp_problem, model)
                if selected_task not in results_per_model:
                    results_per_model[selected_task] = set(), set(), {}
                model_class_matches, model_class_non_matches, pair_confidences_model = results_per_model[selected_task]
                model_class_matches.update(class_match_set)
                model_class_non_matches.update(class_non_match_set)
                pair_confidences_model.update(pair_confidences)
                del unsolved_numpy_dict[unsolved_problem]
            else:
                # if we found no similar cluster, we have to build a new one.
                # print("build new model for cluster from unsolved data")
                unsolved_lps = {}
                unsolved_list = []
                for up in unsolved_cluster:
                    unsolved_list.append((up[0], up[1], data_source_comp[up]))
                unsolved_lps[unsolved_problem] = unsolved_list
                allocated_budget = {unsolved_problem: BUDGET_RETRAIN}
                new_training_data = model_generation.select_new_training_data(allocated_budget, unsolved_lps,
                                                                              data_source_comp,
                                                                              ACTIVE_LEARNING_ITERATION_BUDGET,
                                                                              gold_links,
                                                                              unsupervised_gold_links, ML_MODEL,
                                                                              MODEL_GENERATION_ALGORITHM)
                new_model_dict, train_data_dict = (
                    model_generation.retrain_models(new_training_data, train_data_dict, ML_MODEL, is_fine_tuned=True,
                                                    is_calibrated=False))
                cal_model_dict.update(new_model_dict)
                weight_dict[unsolved_problem] = weights
                class_match_set, class_non_match_set, pair_confidences = active_learning_solution \
                    .classify(data_source_comp[unsolved_problem], cal_model_dict[unsolved_problem][0])
                model_class_matches, model_class_non_matches, pair_confidences_model = results_per_model[
                    unsolved_problem]
                model_class_matches.update(class_match_set)
                model_class_non_matches.update(class_non_match_set)
                pair_confidences_model.update(pair_confidences)
                if IS_RECLUSTER:
                    lp_list = []
                    for up in unsolved_cluster:
                        uns_lp_numpy = statistical_tests.prepare_numpy_to_similarity_comparison_from_lp(
                            data_source_comp[(up[0], up[1])])
                        lp_list.append((up, uns_lp_numpy))
                    # print("add new unsolved problem")
                    selected_tasks[unsolved_problem] = [(up, 0) for up in unsolved_cluster]
                else:
                    selected_numpy_dict[unsolved_problem] = unsolved_numpy_dict[unsolved_problem]
                del unsolved_numpy_dict[unsolved_problem]
        solved_tasks += 1
        # retrain model after calculated number of solved tasks and retrain the models
        result_dictionary[unsolved_problem] = (class_match_set, class_non_match_set, pair_confidences)
    # print("relabeled data {}".format(labeled_data))
    overall_elapsed_time = time.time() - start_overall_time
    # print(f"overall Elapsed time: {overall_elapsed_time:.2f} seconds")
    # ===================================================
    # Step 8: Evaluate the Results
    # ===================================================
    # Evaluate the performance of the labeled tasks

    model_results = evaluate_prediction_per_model(results_per_model, gold_links)
    for task, results in model_results.items():
        print(task)
        print("P={} R={} F1={}".format(results[3], results[4], results[5]))
    tps, fps, fns, p, r, f = evaluate_predictions(result_dictionary, gold_links)
    tps_list.append(tps)
    fps_list.append(fps)
    fn_list.append(fns)
    p_list.append(p)
    r_list.append(r)
    f_list.append(f)
    run_t_list.append(overall_elapsed_time)
data_set_name = ''
if 'dexter' in args.linkage_tasks_dir:
    data_set_name = 'Dexter'
elif 'wdc_almser' in args.linkage_tasks_dir:
    data_set_name = 'WDC-computer'
elif 'music_almser' in args.linkage_tasks_dir:
    data_set_name = 'Music'
with open('results/results.csv', 'a') as result_file:
    result_file.write(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(STATISTICAL_TEST,
                                                                                     FEATURE_CASE,
                                                                                     COMMUNITY_DETECTION_ALGORITHM,
                                                                                     False,
                                                                                     ACTIVE_LEARNING_TOTAL_BUDGET,
                                                                                     used_budgets_unsolved,
                                                                                     0,
                                                                                     IS_RECLUSTER,
                                                                                     UNSOLVED_RATIO,
                                                                                     LP_SELECTION_STRATEGY,
                                                                                     sum(tps_list) + sum(
                                                                                         fn_list),
                                                                                     sum(tps_list),
                                                                                     sum(fps_list),
                                                                                     sum(fn_list),
                                                                                     f"{mean(p_list):.3f}",
                                                                                     f"{mean(r_list):.3f}",
                                                                                     f"{mean(f_list):.3f}",
                                                                                     f"{std(f_list):.3f}",
                                                                                     f"{mean(run_t_list):.3f}",
                                                                                     SELECTION_STRATEGY,
                                                                                     data_set_name,
                                                                                     MODEL_GENERATION_ALGORITHM))
    result_file.close()
