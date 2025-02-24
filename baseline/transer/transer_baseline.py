import argparse
import operator
import os
import random
import time
from statistics import mean

import numpy as np
import pandas as pd
from baseline.transer.model import match_data
import baseline.transer.model


# Define the main path for the project
# MAIN_PATH = '/home/dbs-experiments/PycharmProjects/metadatatransferlearning'

def transform_to_trans_er(solved_problems, gold_links):
    solved_problem_df_list = []
    for source, target, lp in solved_problems:
        columns = []
        features = []
        labels = []
        for k, v in lp.items():
            features.append(v)
            labels.append(tuple(sorted(k)) in gold_links)
        if len(columns) == 0:
            columns = [str(i) for i in range(len(v))]
        numpy_features = np.asarray(features)
        numpy_features[numpy_features == -1] = 0
        df = pd.DataFrame(numpy_features, columns=columns)
        series = pd.Series(labels)
        solved_problem_df_list.append((source, target, {'features': df, 'labels': series}))
    return solved_problem_df_list


def transform_numpy_to_trans_er(lp_numpy, labels):
    columns = [str(i) for i in range(lp_numpy.shape[1])]
    df = pd.DataFrame(np.asarray(lp_numpy), columns=columns)
    series = pd.Series(labels)
    return {'features': df, 'labels': series}


MAIN_PATH = os.getcwd()
fv_splitter = "_"
# Add the path to the custom module directory

# Local application imports
parser = argparse.ArgumentParser(description='rl generation')

parser.add_argument('--data_file', '-d', type=str, default='datasets/dexter/DS-C0/SW_0.3', help='data file')
# parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/wdc_almser/train_pairs_fv.csv',
#                     help='train pairs')
# parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/wdc_almser/test_pairs_fv.csv',
#                     help='test pairs')
# parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/music_almser/train_pairs_fv.csv',
#                     help='train pairs')
# parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/music_almser/test_pairs_fv.csv',
#                     help='test pairs')

# parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/wdc_almser',
#                     help='linkage problem directory')
parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/dexter',
                    help='linkage problem directory')
# parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/music_almser',
#                     help='linkage problem directory')
parser.add_argument('--statistical_test', '-s', type=str, default='ks_test',
                    choices=['ks_test', 'wasserstein_distance', 'calculate_psi'],
                    help='statistical test for comparing lps')
parser.add_argument('--ratio_sim_atomic_dis', '-rs', type=int, default=0,
                    help='amount of similar feature distributions so the lps are considered as similar')
parser.add_argument('--comm_detect', '-cd', type=str, default='leiden',
                    choices=['leiden', 'girvan_newman', 'label_propagation_clustering', 'louvain'],
                    help='communitiy detection algorithm')
parser.add_argument('--relevance_score', '-re', type=str, default='betweenness_centrality',
                    choices=['betweenness_centrality','largest', 'pageRank'],
                    help='relevance score for ordering the linkage problems in a cluster')
parser.add_argument('--active_learning', '-al', type=str, default='almser', choices=['almser', 'bootstrap'],
                    help='active learning algorithm')
parser.add_argument('--min_budget', '-mb', type=int, default=50,
                    help='minimum budget for each cluster')
parser.add_argument('--total_budget', '-tb', type=int, default=2000,
                    help='total budget')
parser.add_argument('--budget_unsolved', '-ub', type=int, default=200,
                    help='budget for unsolved linkage problems being not similar to any solved one')
parser.add_argument('--batch_size', '-b', type=int, default=5,
                    help='batch size')

args = parser.parse_args()

# Define the configuration parameters
ACTIIVE_LEARNING_MIN_BUDGET = 50
ACTIVE_LEARNING_ITERATION_BUDGET = 5
ACTIVE_LEARNING_TOTAL_BUDGET = 10000

data_file = 'datasets/dexter/DS-C0/SW_0.3'
linkage_tasks_dir = 'data/linkage_problems/dexter'
output_path = 'results'
# Active Learning Settings
runs = 3

k = 10
t_c = 0.9
t_l = 0.9
t_p = 0.9
is_al = False

from meta_tl.transfer.incremental.util import split_linkage_problem_tasks, \
    split_linkage_problem_tasks_on_training_data_pairs
from meta_tl.data_io import linkage_problem_io
from record_linkage.classification.machine_learning import constants, active_learning_solution
from record_linkage.evaluation import metrics
from meta_tl.data_io.test_data import reader, wdc_reader, almser_linkage_reader
import os

p_list, r_list, f_list = [], [], []
runtime = []
data_set = ''
for i in range(runs):
    unsupervised_gold_links = set()
    file_name = os.path.join(MAIN_PATH, data_file)
    entities, _, _ = reader.read_data(file_name)
    gold_clusters = reader.generate_gold_clusters(entities)
    gold_links = metrics.generate_links(gold_clusters)
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
        print("tps overall {}".format(len(gold_links)))
    unsupervised_gold_links = set()
    ML_MODEL = constants.RF
    RECORD_LINKAGE_TASKS_PATH = os.path.join(MAIN_PATH, args.linkage_tasks_dir)
    data_source_comp: dict[(str, str):[dict[(str, str):list]]] = linkage_problem_io.read_linkage_problems(
        RECORD_LINKAGE_TASKS_PATH, deduplication=False)
    linkage_problems = [(k[0], k[1], lp) for k, lp in data_source_comp.items()]
    files = [t[0] + "_" + t[1] for t in linkage_problems]
    print("number of lp problems {}".format(len(files)))

    data_set = ''
    if 'dexter' in args.linkage_tasks_dir:
        solved_problems, integrated_sources, unsolved_problems = split_linkage_problem_tasks(linkage_problems,
                                                                                             split_ratio=0.5,
                                                                                             is_shuffle=True)
        data_set = 'Dexter'
    elif 'wdc_computer' in args.linkage_tasks_dir:
        train_check = set(train_tp_links)
        train_check.update(train_tn_links)
        test_check = set(test_tp_links)
        test_check.update(test_tn_links)
        print("test links:{}".format(len(test_check)))
        solved_problems, integrated_sources, unsolved_problems, data_source_comp = (
            split_linkage_problem_tasks_on_training_data_pairs
            (data_source_comp, train_check, test_check))
        removed_pairs = set(data_source_comp.keys()).difference(
            set([(t[0], t[1]) for t in solved_problems]).union([(t[0], t[1]) for t in unsolved_problems]))
        print("removed pairs {}".format(len(removed_pairs)))
        for t in removed_pairs:
            del data_source_comp[t]
        assert len(solved_problems) + len(unsolved_problems) == len(data_source_comp), (str(len(data_source_comp)) +
                                                                                        "  " + str(
                    len(solved_problems) + len(unsolved_problems)))
    elif 'wdc_almser' in args.linkage_tasks_dir or 'music_almser' in args.linkage_tasks_dir:
        if 'wdc_almser' in args.linkage_tasks_dir:
            data_set = 'WDC-computer'
        else:
            data_set = 'Music'
        #data_source_comp = wdc_linkage_reader.split_linkage_problems(args.train_pairs, args.test_pairs, data_source_comp)
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
        print("number of tps in lps {}".format(tps_check))
    largest_problems = solved_problems
    print("solved lps {}".format(len(solved_problems)))
    print("unsolved {}".format(len(unsolved_problems)))

    transer_unsolved_list = transform_to_trans_er(unsolved_problems, gold_links)
    target_training_data = []
    start_time = time.time()
    print("start_time: " + str(start_time))
    if is_al:
        target_problem = dict()
        all_features = 0
        for lp_tuple in largest_problems:
            all_features += len(lp_tuple[2])
            target_problem.update(lp_tuple[2])
        assert len(target_problem) == all_features
        al = active_learning_solution.ActiveLearningBootstrap(ACTIVE_LEARNING_TOTAL_BUDGET, ACTIVE_LEARNING_ITERATION_BUDGET)
        training_data, class_labels = al.select_training_data(target_problem, gold_links)
        integrated_dict_data = transform_numpy_to_trans_er(training_data, class_labels)
        target_training_data.append(("all", "all", integrated_dict_data))
    else:
        transer_list = transform_to_trans_er(largest_problems, gold_links)
        feature_frames = []
        label_series = []
        for lp_tuple in transer_list:
            feature_frames.append(lp_tuple[2]['features'])
            label_series.append(lp_tuple[2]['labels'])
        integrated_dict_data = {'features': pd.concat(feature_frames, ignore_index=True),
                                'labels': pd.concat(label_series, ignore_index=True)}

    index = 0
    result_list = []
    while len(transer_unsolved_list) > 0:
        lp_tuple = transer_unsolved_list[index % len(transer_unsolved_list)]
        if lp_tuple[0].replace('_test', '') in integrated_sources or lp_tuple[1].replace('_test', '') in integrated_sources:
            #target_tuple = random.choice(target_training_data)
            tmp_result_list = match_data(integrated_dict_data['features'], integrated_dict_data['labels'],
                                         lp_tuple[2]['features'], lp_tuple[2]['labels'], k,
                                         t_c, t_l, t_p)
            result_list.extend(tmp_result_list)
            print(tmp_result_list)
            transer_unsolved_list.pop(index % len(transer_unsolved_list))
            print("unsolved:" + str(len(transer_unsolved_list)))
        index += 1

    elapsed_time = time.time() - start_time
    runtime.append(elapsed_time)
    tp = sum([r['tp'] for r in result_list])
    fp = sum([r['fp'] for r in result_list])
    fn = sum([r['fn'] for r in result_list])
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print("overall p:{}r:{}f1{}".format(p, r, f1))
    p_list.append(p)
    print(elapsed_time)
    r_list.append(r)
    f_list.append(f1)

with open('results/trans_er_baseline.csv', 'a') as result_file:
    result_file.write("{},{},{},{},{},{},{},{},{}\n".format(ACTIVE_LEARNING_TOTAL_BUDGET if is_al else -1,
                                                         t_c,
                                                         t_l,
                                                         t_p,
                                                         mean(p_list),
                                                         mean(r_list),
                                                         mean(f_list),
                                                         np.std(f_list),
                                                         mean(runtime), data_set))
