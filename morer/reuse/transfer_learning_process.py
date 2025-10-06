import sys

from record_linkage.classification.machine_learning import active_learning_solution
from utils import *
import xgboost as xgb


def transfer_learning_process_main(selected_linkage_tasks_from_communities, model_community_dict, lp_problems):
    linkage_result = {}
    print(model_community_dict.keys())
    for selected_linkage_task, coressponding_linkage_tasks in selected_linkage_tasks_from_communities.items():
        model = model_community_dict[selected_linkage_task]
        # the baseline is also evaluated with the linkage task.
        lp_problem: dict[(str, str):list] = lp_problems[selected_linkage_task]
        class_match_set, class_nonmatch_set, confidence_pair = active_learning_solution.classify(lp_problem, model)
        linkage_result[selected_linkage_task] = (class_match_set, class_nonmatch_set, confidence_pair)
        print(f"Inference on the coresponding tasks!")
        for coressponding_linkage_task in coressponding_linkage_tasks:
            lp_problem: dict[(str, str):list] = lp_problems[coressponding_linkage_task[0]]
            class_match_set, class_nonmatch_set, confidence_pair = active_learning_solution.classify(lp_problem, model)
            linkage_result[coressponding_linkage_task[0]] = (class_match_set, class_nonmatch_set, confidence_pair)
    return linkage_result
