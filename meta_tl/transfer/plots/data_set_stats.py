import argparse
import os

import pandas as pd

from meta_tl.data_io import linkage_problem_io
from meta_tl.data_io.test_data import almser_linkage_reader, reader
from record_linkage.evaluation import metrics

parser = argparse.ArgumentParser(description='rl generation')
parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/wdc_almser',
                    help='linkage problem directory')
# parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/dexter',
#                     help='linkage problem directory')
# parser.add_argument('--linkage_tasks_dir', '-l', type=str, default='data/linkage_problems/music_almser',
#                     help='linkage problem directory')
parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/wdc_almser/train_pairs_fv.csv',
                    help='train pairs')
parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/wdc_almser/test_pairs_fv.csv',
                    help='test pairs')
# parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/music_almser/train_pairs_fv.csv',
#                     help='train pairs')
# parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/music_almser/test_pairs_fv.csv',
#                     help='test pairs')

MAIN_PATH = os.getcwd()
args = parser.parse_args()
train_tp_links, train_tn_links, test_tp_links, test_tn_links, unsup_train_tp_links, unsup_train_tn_links = (
            almser_linkage_reader.read_wdc_links(args.train_pairs, args.test_pairs))

stat = pd.DataFrame(columns=["name", "# linkage problems", "# record pairs", "# Matches"])
name_list, linkage_p, rp_list, matches = [],[],[],[]
for ds in ['data/linkage_problems/dexter', 'data/linkage_problems/wdc_almser', 'data/linkage_problems/music_almser']:
    RECORD_LINKAGE_TASKS_PATH = os.path.join(MAIN_PATH, ds)
    data_source_comp: dict[(str, str):[dict[(str, str):list]]] = linkage_problem_io.read_linkage_problems(
        RECORD_LINKAGE_TASKS_PATH, deduplication=False)
    linkage_problems = [(k[0], k[1], lp) for k, lp in data_source_comp.items()]
    num_lps = len(linkage_problems)
    all_links = sum(len(lp[2]) for lp in linkage_problems)
    name = ""
    if 'dexter' in ds:
        file_name = os.path.join(MAIN_PATH, 'datasets/dexter/DS-C0/SW_0.3')
        entities, _, _ = reader.read_data(file_name)
        gold_clusters = reader.generate_gold_clusters(entities)
        gold_links = metrics.generate_links(gold_clusters)
        num_gold_links = len(gold_links)
        name = 'Dexter'
    elif 'wdc_almser' in ds:
        train_tp_links, train_tn_links, test_tp_links, test_tn_links, unsup_train_tp_links, unsup_train_tn_links = (
            almser_linkage_reader.read_wdc_links('data/linkage_problems/wdc_almser/train_pairs_fv.csv',
                                                 'data/linkage_problems/wdc_almser/test_pairs_fv.csv'))
        num_lps = len(linkage_problems)
        num_gold_links = len(train_tp_links) + len(test_tp_links)
        if 'wdc_almser' in ds:
            name = 'WDC-computer'
        else:
            name = 'Music'
    elif 'music_almser' in ds:
        train_tp_links, train_tn_links, test_tp_links, test_tn_links, unsup_train_tp_links, unsup_train_tn_links = (
            almser_linkage_reader.read_wdc_links('data/linkage_problems/music_almser/train_pairs_fv.csv',
                                                 'data/linkage_problems/music_almser/test_pairs_fv.csv'))
        num_lps = len(linkage_problems)
        num_gold_links = len(train_tp_links) + len(test_tp_links)
    name_list.append(name)
    linkage_p.append(num_lps)
    rp_list.append(all_links)
    matches.append(num_gold_links)
data_dict = {"name":name_list,
             "# linkage problems":linkage_p,
             "# record pairs":rp_list,
             "# matches":matches}

data_frame = pd.DataFrame(data=data_dict)
print(data_frame.to_latex(index=False, escape=True))


stat_table = pd.read_csv('results/lp_ratio_comparison.csv')
reduced_table = stat_table[['data_set', 'al', 'lp_ratio', 'BUDGET', 'F']]
sorted_df = reduced_table.sort_values(by=['data_set', 'BUDGET', 'al'], ascending=True)
print(sorted_df.to_latex(index=False, escape=True))



