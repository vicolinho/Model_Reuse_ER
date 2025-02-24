import random


# Util module to split linkage problems

def split_linkage_problem_tasks(linkage_problems:list[(str,str,dict[(str,str):list])], split_ratio, is_shuffle):
    """
    params:
            linkage_problems: list of pairwise linkage problems
            split_ratio: ratio of the number of problems being solved
            is_shuffle: shuffle the unsolved problems to simulate the uncertainty
    """
    solved_index = round(len(linkage_problems) * split_ratio)
    solved_lps = linkage_problems[:solved_index]
    to_merge_problems = linkage_problems[solved_index:]
    integrated_data_sources = set([t[0] for t in solved_lps]).union(set([t[1] for t in solved_lps]))
    if is_shuffle:
        random.seed(37)
        random.shuffle(to_merge_problems)
    return solved_lps, integrated_data_sources, to_merge_problems


def split_linkage_problem_tasks_on_training_data_pairs(linkage_problems:dict[(str,str):dict[(str,str):list]],
                                                       train_pairs, test_pairs):
    solved_lps = {}
    to_merge_problems = {}
    integrated_data_sources = set()

    for (s, t), lp in linkage_problems.items():
        for pair, vec in lp.items():
            if pair in train_pairs:
                if (s, t) not in solved_lps:
                    solved_lps[(s,t)] = {}
                sim_vecs = solved_lps[(s, t)]
                sim_vecs[pair] = vec
                integrated_data_sources.add(s)
                integrated_data_sources.add(t)
            if pair in test_pairs:
                if (s, t) not in to_merge_problems:
                    to_merge_problems[(s, t)] = {}
                sim_vecs = to_merge_problems[(s, t)]
                sim_vecs[pair] = vec
    solved_lps_list = [(k[0], k[1], v) for k,v in solved_lps.items() if len(v) > 10]
    to_merge_problems_list = [(k[0], k[1], v) for k, v in to_merge_problems.items()]
    intersection = set(solved_lps.keys()).intersection(set(to_merge_problems.keys()))
    pair_count = 0
    for t in to_merge_problems_list:
        pair_count += len(t[2])
    print("solved problems: " + str(len(solved_lps_list)))
    print("to merge problems: " + str(len(to_merge_problems_list)))
    print("test pairs: " + str(pair_count))
    print(len(intersection))
    modified_solved_lps_list = []
    for lp in solved_lps_list:
        if (lp[0], lp[1]) in intersection:
            solved = (lp[0] + "_train", lp[1] + "_train")
            linkage_problems[solved] = lp[2]
            linkage_problems[(lp[0], lp[1])] = to_merge_problems[(lp[0], lp[1])]
        else:
            solved = (lp[0] + "_train", lp[1] + "_train")
            del linkage_problems[(lp[0], lp[1])]
            linkage_problems[solved] = lp[2]
        modified_solved_lps_list.append((lp[0] + "_train", lp[1] + "_train", lp[2]))
    return modified_solved_lps_list, integrated_data_sources, to_merge_problems_list, linkage_problems

