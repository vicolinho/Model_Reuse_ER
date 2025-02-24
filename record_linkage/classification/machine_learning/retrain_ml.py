from record_linkage.classification.machine_learning import constants, active_learning_solution
from record_linkage.classification.machine_learning.active_learning_solution import ActiveLearningBootstrap


def train(rec_a_dict, rec_b_dict, weight_vec_dict, model='', ml_type=constants.RF, gold_labels=set(), **kwargs):
    trained_model = None
    if ml_type == 'AL':
        al = ActiveLearningBootstrap(kwargs[active_learning_solution.BUDGET],
                                     kwargs[active_learning_solution.ITER_BUDGET],
                                     kwargs[active_learning_solution.k])
        trained_model = al.select_training_data(weight_vec_dict, gold_labels, model)
    return trained_model