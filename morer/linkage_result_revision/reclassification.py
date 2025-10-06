from record_linkage.classification.machine_learning import threshold_classification_solution
from record_linkage.classification.machine_learning import active_learning_solution


def reclassify_pairs(reclassified_pairs, method:str, **kwargs):
    if method == 't_avg':
        class_match_set, class_non_match_set, _ = threshold_classification_solution.threshold_classify(reclassified_pairs, kwargs["t"])
    elif method == 't_min':
        class_match_set, class_non_match_set, _ = threshold_classification_solution.min_threshold_classify(
            reclassified_pairs, kwargs["t"])
    elif method == 't_weight':
        class_match_set, class_non_match_set, _ = threshold_classification_solution.weighted_similarity_classify(
            reclassified_pairs, kwargs['weight'], kwargs["t"])
    elif method == 'ml':
        class_match_set, class_non_match_set, _ = active_learning_solution.classify(reclassified_pairs, kwargs['model'])

