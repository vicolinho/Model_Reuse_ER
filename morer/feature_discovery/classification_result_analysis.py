import operator
from math import ceil

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from morer.quality_estimation import quality_estimation


def determine_false_positives(class_match_set: set[tuple[str,str]], weight_dict: dict[(str, str):float], is_dirty=False,
                              is_plot=False, true_match_set=None):
    """
    Estimate the number of true positives and determines the potential false positives using the estimated precision
    Parameter Description:
       class_match_set : set of classified record pairs
       weight_dict: dictionary with aggregated similarities for each record pair
       is_dirty: flag for the type of data_io sources. is_dirty=true means that the data_io sources can potentially contain
       intra duplicates
    """
    if not is_dirty:
        estimated_prec, tp_scores = quality_estimation.evaluate(class_match_set, weight_dict)
    else:
        estimated_prec, tp_scores = quality_estimation.evaluate_duplicate(class_match_set, weight_dict)
    est_fp = (1 - estimated_prec) * len(class_match_set)
    print("estimated precision: {} potential {} false positives".format(estimated_prec, est_fp))
    sorted_scores = sorted(tp_scores.items(), key=operator.itemgetter(1))
    fp_candidates = []
    scores = []
    index = 0
    for p, score in sorted_scores:
        if index < est_fp:
            t = p
            if p not in weight_dict:
                t = tuple((p[1], p[0]))
            fp_candidates.append(t)
            class_match_set.remove(t)
        if is_plot:
            scores.append((score, t in true_match_set))
        index += 1
    if is_plot:
        tp_data_frame = pd.DataFrame(data=scores, columns=["score", "TP"])
        sns.histplot(data=tp_data_frame, x='score', hue='TP', multiple='dodge')
        plt.show()
    return fp_candidates, class_match_set
