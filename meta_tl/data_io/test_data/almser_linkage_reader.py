import os
import sys

import pandas as pd
import argparse

from meta_tl.data_io import linkage_problem_io


def read_linkage_features(linkage_dir):
    data_comp = {}
    total_links = set()
    for file in os.listdir(linkage_dir):
        sources = file.split('_')
        s = sources[0]
        t = sources[1].replace('.csv', '')
        linkage_frame = pd.read_csv(os.path.join(linkage_dir, file))
        id_lists = list(zip(linkage_frame['source_id'].to_list(), linkage_frame['target_id'].to_list()))
        total_links = total_links.union(set(id_lists))
        labels = linkage_frame['label'].to_list()
        gold_links = set()
        for i in range(len(id_lists)):
            if labels[i]:
                gold_links.add(tuple(sorted(id_lists[i])))

        features = linkage_frame.iloc[:, 4:].values

        linkage_problem = {tuple(sorted(id_lists[i])): features[i] for i in range(len(id_lists))}
        data_comp[(s, t)] = linkage_problem
        print("pair {}".format((s, t)))
        print(len(linkage_problem))
        print(len(gold_links))
    print(len(total_links))
    return data_comp


def read_wdc_links(train_labels, test_data):
    linkage_frame = pd.read_csv(train_labels)
    id_lists = list(zip(linkage_frame['source_id'].to_list(), linkage_frame['target_id'].to_list()))
    train_tp_links = set()
    train_tn_links = set()
    unsupervised_train_tp_links = set()
    unsupervised_train_tn_links = set()
    labels = linkage_frame['label'].to_list()
    unsupervised_labels = linkage_frame['unsupervised_label'].to_list()
    for i in range(len(id_lists)):
        if labels[i]:
            train_tp_links.add(tuple(sorted(id_lists[i])))
        else:
            train_tn_links.add(tuple(sorted(id_lists[i])))
        if unsupervised_labels[i]:
            unsupervised_train_tp_links.add(tuple(sorted(id_lists[i])))
        else:
            unsupervised_train_tn_links.add(tuple(sorted(id_lists[i])))
    linkage_frame = pd.read_csv(test_data)
    id_lists = list(zip(linkage_frame['source_id'].to_list(), linkage_frame['target_id'].to_list()))
    labels = linkage_frame['label'].to_list()
    test_tp_links = set()
    test_tn_links = set()
    for i in range(len(id_lists)):
        if labels[i]:
            test_tp_links.add(tuple(sorted(id_lists[i])))
        else:
            test_tn_links.add(tuple(sorted(id_lists[i])))
    return train_tp_links, train_tn_links, test_tp_links, test_tn_links, unsupervised_train_tp_links, unsupervised_train_tn_links


def split_linkage_problems(train_file, test_file, data_comp: dict[(str, str):dict]):
    train_tps_links, train_tns_links, test_tp_links, test_tn_links, _, _ = read_wdc_links(train_file, test_file)
    print("# matches in train: {}".format(len(train_tps_links)))
    print("# non-matches in train: {}".format(len(train_tns_links)))
    print("# matches in test: {}".format(len(test_tp_links)))
    print("# non-matches in test: {}".format(len(test_tn_links)))
    modified_linkage_problems = {}
    all_train_links = train_tps_links.union(train_tns_links)
    all_test_links = test_tp_links.union(test_tn_links)
    all_source_links = set()
    for (s, t), links in data_comp.items():
        for p, sims in links.items():
            all_source_links.add(p)
            if tuple(sorted(p)) in all_train_links:
                if (s + "_train", t + "_train") not in modified_linkage_problems:
                    modified_linkage_problems[(s + "_train", t + "_train")] = dict()
                new_sims = modified_linkage_problems[(s + "_train", t + "_train")]
                new_sims[tuple(sorted(p))] = sims
            elif tuple(sorted(p)) in all_test_links:
                if (s + "_test", t + "_test") not in modified_linkage_problems:
                    modified_linkage_problems[(s + "_test", t + "_test")] = dict()
                new_sims = modified_linkage_problems[(s + "_test", t + "_test")]
                new_sims[tuple(sorted(p))] = sims
            else:
                pass
                # print("not in both {}".format(p))
    for l in all_train_links:
       if l not in all_source_links:
           print("missing train vector {}".format(l))
    for l in all_test_links:
       if l not in all_source_links:
           print("missing test vector {}".format(l))

    return modified_linkage_problems


if __name__ == '__main__':
    parser = argparse.ArgumentParser("almser linkage problem generation")
    parser.add_argument('--feature_vector_pair_folder', '-ff', type=str, default='data/linkage_problems/music_almser/source_pairs',
                        help='folder for feature vectors for each source pair')
    parser.add_argument('--train_pairs', '-tp', type=str, default='data/linkage_problems/music_almser/train_pairs_fv.csv',
                        help='train pairs')
    parser.add_argument('--test_pairs', '-gs', type=str, default='data/linkage_problems/music_almser/test_pairs_fv.csv',
                        help='test pairs')
    parser.add_argument('--linkage_output', '-lo', type=str, default='data/linkage_problems/music_almser',
                        help='path for linkage problem output')
    args = parser.parse_args()
    folder = os.path.join(os.getcwd(), args.feature_vector_pair_folder)
    data_comp = read_linkage_features(folder)
    mod_data_comp = split_linkage_problems(args.train_pairs, args.test_pairs, data_comp)
    linkage_problem_io.dump_linkage_problems(mod_data_comp, args.linkage_output)
