import operator

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, KFold
from tqdm import tqdm

from record_linkage.classification.machine_learning import constants, farthest_first_selection

BUDGET = "budget"

ITER_BUDGET = "iter_budget"

k = "k"


class ActiveLearningBootstrap:

    def __init__(self, budget=1000, iteration_budget=20, k=50):
        '''

        :param budget: total labelling budget
        :param iteration_budget: number of pairs being labelled in each iteration
        :param k: number of classifiers generated by bootstrapping
        '''
        self.budget = budget
        self.iteration_budget = iteration_budget
        self.k = k

    def select_training_data(self, sim_vec_dict, true_match_set, record_pair_scoring_inter={}, record_pair_scoring_intra={}):
        num_train_rec = len(sim_vec_dict)
        num_features = len(list(sim_vec_dict.values())[0])

        print('  Number of training records and features: %d / %d' % \
              (num_train_rec, num_features))

        # all similarity vectors
        all_train_data = np.zeros([num_train_rec, num_features])
        # class label 0 and 1 for each vector in all_train_data
        all_train_class = np.zeros(num_train_rec)
        inter_scoring = np.ones(num_train_rec)
        intra_scoring = np.ones(num_train_rec)

        rec_pair_id_list = []

        num_pos = 0
        num_neg = 0
        i = 0
        # initialization of numpy arrays representing the similarity vectors and the corresponding classes
        for (rec_id1, rec_id2) in sim_vec_dict:
            rec_pair_id_list.append((rec_id1, rec_id2))
            sim_vec = sim_vec_dict[(rec_id1, rec_id2)]
            if len(record_pair_scoring_inter) > 0:
                inter_scoring[i] = record_pair_scoring_inter[(rec_id1, rec_id2)]
            if len(record_pair_scoring_intra) > 0:
                intra_scoring[i] = record_pair_scoring_intra[(rec_id1, rec_id2)]

            all_train_data[:][i] = sim_vec
            if tuple(sorted((rec_id1, rec_id2))) in true_match_set:
                all_train_class[i] = 1.0
                num_pos += 1
            else:
                all_train_class[i] = 0.0
                num_neg += 1
            i += 1
        score_features = np.stack((inter_scoring, intra_scoring, all_train_class), axis=-1)
        print(score_features.shape)
        data_Frame = pd.DataFrame(score_features, columns=["inter_score", "intra_score", "all_train_class"])
        print('  Number of positive and negative records: %d / %d' % \
              (num_pos, num_neg))
        print('')
        # initial training data_io for active learning method
        class_sum = 0
        if self.iteration_budget > self.budget:
            iteration_budget = self.budget
        else:
            iteration_budget = self.iteration_budget
        while class_sum == 0:
            seed_index = farthest_first_selection.graipher(all_train_data, iteration_budget)
            # seed_index = np.random.choice(all_train_class.shape[0], self.iteration_budget, replace=False)
            current_train_vectors = all_train_data[seed_index]
            current_train_class = all_train_class[seed_index]
            class_sum = np.sum(current_train_class)
        # remove selected vectors and classes from original vectors resp. classes
        unlabeled_vectors = np.delete(all_train_data, seed_index, axis=0)
        unlabeled_classes = np.delete(all_train_class, seed_index, axis=0)
        # set the number of used budget to the number of selected seed vectors
        used_budget = current_train_vectors.shape[0]
        # iterative active learning method
        for current_budget in tqdm(range(used_budget, self.budget, iteration_budget)):
            if unlabeled_vectors.shape[0] == 0:
                break
            if iteration_budget > self.budget - current_budget:
                iteration_budget = self.budget - current_budget
            else:
                iteration_budget = self.iteration_budget
            # generate k models based on bootstrapping
            classifiers = []
            # ADD classifier generation code here
            for i in range(self.k):
                model = sklearn.tree.DecisionTreeClassifier()
                bootstrapping_indices = np.random.choice(current_train_vectors.shape[0], current_train_vectors.shape[0],
                                                         replace=True)

                subset_vectors = current_train_vectors[bootstrapping_indices]
                subset_classes = current_train_class[bootstrapping_indices]
                model.fit(subset_vectors, subset_classes)
                classifiers.append(model)

            # compute the uncertainty for each similarity vector in unlabelled_vectors
            # result should be a dictionary <index of unlabeled_vector, uncertainty value of the corresponding vector>
            # ADD uncertainty computation code here
            distribution = np.zeros((unlabeled_vectors.shape[0], self.k))
            for i in range(len(classifiers)):
                model_i = classifiers[i]
                classes = model_i.predict(unlabeled_vectors)
                distribution[:, i] = classes
            agg_distribution = distribution.sum(1)

            uncertainties = {}
            for i in range(len(distribution)):
                x_u = agg_distribution[i] / len(classifiers)
                # uncertainties[i] = x_u * (1 - x_u)
                # TODO check if the scoring works better
                uncertainties[i] = ((x_u * (1 - x_u) * 4) + inter_scoring[i])/2
            # END uncertainty computation
            # sort the unlabelled vectors by the computed uncertainty and select new vectors
            candidate_examples = sorted(uncertainties.items(), key=operator.itemgetter(1), reverse=True)[
                                 :min(iteration_budget, len(uncertainties))]
            next_batch_idxs = [val[0] for val in candidate_examples]
            new_vectors = unlabeled_vectors[next_batch_idxs]
            new_classes = unlabeled_classes[next_batch_idxs]
            # remove selected vectors and classes from the unlabelled vectors resp. classes
            unlabeled_vectors = np.delete(unlabeled_vectors, next_batch_idxs, axis=0)
            unlabeled_classes = np.delete(unlabeled_classes, next_batch_idxs, axis=0)
            inter_scoring = np.delete(inter_scoring, next_batch_idxs, axis=0)
            # add the selected vectors and classes to the existing training data_io set
            current_train_vectors = np.vstack((current_train_vectors, new_vectors))
            current_train_class = np.hstack((current_train_class, new_classes))
            # increase the used budget
            # used_budget = current_train_vectors.shape[0]
        # train decision tree using the generated training data_io set
        print("postive ratio: {}".format(current_train_class.sum()/current_train_class.shape[0]))
        return current_train_vectors, current_train_class


def train_model(current_train_vectors: np.ndarray, current_train_class: np.ndarray, model_name, is_calibrated=True,
                is_fine_tuned=False):
    if model_name == constants.DECISION_TREE:
        model = sklearn.tree.DecisionTreeClassifier()
        criterion = ['gini', 'entropy']
        max_depth = [2, 4, 6, 8, 10, 12]
        min_samples_split = [2, 6, 10]  # minimum sample number to split a node
        min_samples_leaf = [1, 3, 4]  # minimum sample number that can be stored in a leaf node
        random_grid = {'criterion': criterion,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf
                       }
    elif model_name == constants.RF:
        n_estimators = [int(x) for x in np.linspace(start=1, stop=20, num=20)]  # number of trees in the random forest
        max_features = ['sqrt']  # number of features in consideration at every split
        max_depth = [int(x) for x in
                     np.linspace(2, 20, num=12)]  # maximum number of levels allowed in each decision tree
        min_samples_split = [2, 6, 10]  # minimum sample number to split a node
        min_samples_leaf = [1, 3, 4]  # minimum sample number that can be stored in a leaf node
        bootstrap = [True, False]  # method used to sample data points
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        model = RandomForestClassifier(max_depth=3)

    # scores = cross_val_score(model, features, labels, cv=5, scoring='f1_macro')

    # print('Best Hyperparameters: %s' % result.best_params_)

    if current_train_class.sum() < 2:
        calibrated_clf = model
        return calibrated_clf.fit(current_train_vectors, current_train_class)
    if is_calibrated:
        calibrated_clf = CalibratedClassifierCV(model, cv=2)
        print("labelled samples:{}".format(current_train_class.shape[0]))
        calibrated_clf.fit(current_train_vectors, current_train_class)
        return calibrated_clf
    elif not is_fine_tuned:
        calibrated_clf = model
        return calibrated_clf.fit(current_train_vectors, current_train_class)
    else:
        cv = KFold(n_splits=3)
        rf_random = RandomizedSearchCV(model, random_grid, n_iter=100, cv=cv, scoring='f1',
                                       verbose=0, random_state=35, n_jobs=-1)
        result = rf_random.fit(current_train_vectors, current_train_class)
        print('Best Score: %s' % result.best_score_)
        return result.best_estimator_, result.best_score_



def classify(sim_vec_dict, model):
    '''
    classify similartiy feature vectors if the given model
    If the model is calibrated, it also returns the probabilities
    Returns:
        class_match_Set: record pairs classified as matches
        class_nonmatch_set: record pairs classified as non-matches
        pair confidence list: list of probabilities for each record pair to be a match
    '''
    num_train_rec = len(sim_vec_dict)
    num_features = len(list(sim_vec_dict.values())[0])
    all_train_data = np.zeros([num_train_rec, num_features])
    rec_pair_id_list = []
    i = 0
    for (rec_id1, rec_id2) in sim_vec_dict:
        rec_pair_id_list.append((rec_id1, rec_id2))
        sim_vec = sim_vec_dict[(rec_id1, rec_id2)]

        all_train_data[:][i] = sim_vec
        i += 1
    predictions = model.predict(all_train_data)
    probabilities = model.predict_proba(all_train_data)
    class_match_set = set()
    class_nonmatch_set = set()
    confidence_pair = {}
    for i in range(len(sim_vec_dict)):
        rec_id_pair = rec_pair_id_list[i]
        if probabilities.shape[1] == 2:
            prob = probabilities[i][1]
        else:
            prob = probabilities[i]
        if predictions[i] == 1:
            class_match_set.add(rec_id_pair)
            confidence_pair[rec_id_pair] = prob
        else:
            class_nonmatch_set.add(rec_id_pair)
    return class_match_set, class_nonmatch_set, confidence_pair
