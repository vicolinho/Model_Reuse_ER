import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from record_linkage.classification.machine_learning import farthest_first_selection


def boost_training_data(training_data: np.ndarray, labels, lp: np.ndarray, lp_problem:dict[(str,str):list],
                        debug_ground_truth):
    print(lp.shape)
    # unique_lp, indices = np.unique(lp, axis=0, return_index=True)
    # unique_debug_ground_truth = debug_ground_truth[indices]
    #new_training_labels = new_training_labels[indices]
    boost_unlabeled_data, unsupervised_labels = informative_selection_edge_wise(training_data, labels, lp,
                                                                                debug_ground_truth)
    if boost_unlabeled_data is not None:
        reverse_vec_rec_pairs = {}
        for k, v in lp_problem.items():
            if tuple(v) not in reverse_vec_rec_pairs:
                reverse_vec_rec_pairs[tuple(v)] = set()
            pairs = reverse_vec_rec_pairs[tuple(v)]
            pairs.add(k)
        unlabeled_class_match = set()
        unlabeled_non_class_match = set()
        new_labels = []
        new_features = []
        pos_features = []
        neg_features = []
        processed_vectors = set()
        for i in range(boost_unlabeled_data.shape[0]):
            vector = boost_unlabeled_data[i].tolist()
            if tuple(vector) not in processed_vectors:
                if tuple(vector) in reverse_vec_rec_pairs:
                    pairs = reverse_vec_rec_pairs[tuple(vector)]
                    label = unsupervised_labels[i]
                    # if label:
                    #     unlabeled_class_match.union(pairs)
                    # else:
                    #     unlabeled_non_class_match.union(pairs)
                for p in pairs:
                    new_features.append(vector)
                    if label:
                        pos_features.append(vector)
                        new_labels.append(1)
                    else:
                        neg_features.append(vector)
                        new_labels.append(0)
                        # if p in lp_problem:
                        #     del lp_problem[p]
                processed_vectors.add(tuple(vector))
        class_ratio = np.sum(labels) / labels.shape[0]
        number = math.ceil((1 - class_ratio) * 1 / class_ratio)
        new_pos_features = []
        new_pos_labels = []
        new_neg_features = []
        new_neg_labels = []
        print("pos/neg unlabelled: {}".format(len(pos_features) / (len(neg_features) + len(pos_features))))
        if class_ratio < 0.5:
            while len(pos_features) > 0:
                if number <= len(neg_features):
                    new_pos_features.append(pos_features.pop(0))
                    new_pos_labels.append(1)
                    for i in range(number):
                        new_neg_features.append(neg_features.pop(0))
                        new_neg_labels.append(0)
                else:
                    break
        else:
            while len(neg_features) > 0:
                if number <= len(pos_features):
                    new_neg_features.append(neg_features.pop(0))
                    new_neg_labels.append(0)
                    for i in range(number):
                        new_pos_features.append(pos_features.pop(0))
                        new_pos_labels.append(1)
                else:
                    break

        print("pos/neg ratio: {}, samples{}".format(class_ratio, labels.shape[0]))

        if len(new_pos_features) > 0 and len(new_neg_features) > 0:

            new_training_data = np.vstack((np.asarray(new_pos_features), np.asarray(new_neg_features)))
            new_training_labels = np.hstack((np.asarray(new_pos_labels), np.asarray(new_neg_labels)))
            new_training_data = np.vstack((training_data, new_training_data))
            new_training_labels = np.hstack((labels, new_training_labels))
            return new_training_data, new_training_labels
    return training_data, labels




def informative_selection_edge_wise(training_feature_edges, labeled_vectors,
                                    unlabelled_feature_edges: np.ndarray,
                                    unlabelled_classes: np.ndarray):
    pos_indices = np.where(labeled_vectors == 1)[0]
    neg_indices = np.where(labeled_vectors == 0)[0]

    # compute entropy based measure

    pos_vectors = training_feature_edges[pos_indices]
    neg_vectors = training_feature_edges[neg_indices]
    cos_diff = cosine_similarity(pos_vectors, neg_vectors)
    cos_same_pos = cosine_similarity(pos_vectors, pos_vectors)
    cos_same_neg = cosine_similarity(neg_vectors, neg_vectors)
    sum_pos_same = np.sum(cos_same_pos, axis=1)
    sum_neg_same = np.sum(cos_same_neg, axis=1)
    sum_pos_diff = np.sum(cos_diff, axis=1)
    sum_neg_diff = np.sum(cos_diff, axis=0)
    entropy_pos = - (
            sum_pos_same / training_feature_edges.shape[0] * np.log2(sum_pos_same / training_feature_edges.shape[0]) \
            + sum_pos_diff / training_feature_edges.shape[0] * np.log2(
        sum_pos_diff / training_feature_edges.shape[0]))
    assert pos_vectors.shape[0] == entropy_pos.shape[0], "wrong dimensions of pos vectors"
    entropy_neg = - (
            sum_neg_same / training_feature_edges.shape[0] * np.log2(sum_neg_same / training_feature_edges.shape[0]) \
            + sum_neg_diff / training_feature_edges.shape[0] * np.log2(
        sum_neg_diff / training_feature_edges.shape[0]))
    # compute uncertainty
    pos_max_sims = np.amax(cos_diff, axis=1)
    neg_max_sims = np.amax(cos_diff, axis=0)
    pos_max_sims = np.reshape(pos_max_sims, (pos_max_sims.shape[0], 1))
    neg_max_sims = np.reshape(neg_max_sims, (neg_max_sims.shape[0], 1))
    s_same_pos = cos_same_pos >= pos_max_sims
    s_same_neg = cos_same_neg >= neg_max_sims
    unc_pos = np.float32(s_same_pos.sum(axis=1))
    unc_neg = np.float32(s_same_neg.sum(axis=1))
    # unc_pos = np.reciprocal(unc_pos)
    # unc_neg = np.reciprocal(unc_neg)
    unc_pos = unc_pos / pos_vectors.shape[0]
    unc_neg = unc_neg / neg_vectors.shape[0]
    info_pos = (unc_pos + entropy_pos) / 2.
    info_neg = (unc_neg + entropy_neg) / 2.
    union = np.hstack((info_pos, info_neg))
    threshold = np.percentile(union, 90)
    pos_indices = np.where(info_pos >= threshold)
    neg_indices = np.where(info_neg >= threshold)

    # select informative training data points
    i_pos = pos_vectors[pos_indices]
    i_pos_thresh = pos_max_sims[pos_indices] + (1 - pos_max_sims[pos_indices]) / 2
    i_neg = neg_vectors[neg_indices]
    i_neg_thresh = neg_max_sims[neg_indices] + (1 - neg_max_sims[neg_indices]) / 2
    next_indices_pos = [[]]
    if i_pos.shape[0] > 0:
        cos_sim_with_unlabelled_positive = cosine_similarity(i_pos, unlabelled_feature_edges)
        selected_unlabelled_pos = cos_sim_with_unlabelled_positive >= i_pos_thresh
        next_indices_pos = np.where(np.any(selected_unlabelled_pos, axis=0))
    next_indices_neg = [[]]
    if i_neg.shape[0] > 0:
        cos_sim_with_unlabelled_negative = cosine_similarity(i_neg, unlabelled_feature_edges)
        selected_unlabelled_neg = cos_sim_with_unlabelled_negative >= i_neg_thresh
        next_indices_neg = np.where(np.any(selected_unlabelled_neg, axis=0))
    # select unlabelled candidates
    new_classes = np.hstack((np.ones(len(next_indices_pos[0])), np.zeros(len(next_indices_neg[0]))))

    ind_vecs = np.hstack((next_indices_pos[0], next_indices_neg[0]))
    debug_labels = unlabelled_classes[ind_vecs.astype(int)]
    agreement = new_classes == debug_labels
    new_vectors = None
    if ind_vecs.shape[0] > 0:
        new_vectors = unlabelled_feature_edges[ind_vecs.astype(int)]
        #farthest first selection
        # if new_vectors.shape[0] > round(1/3.0*training_feature_edges.shape[0]):
        #     boost_number = round(1/3.0*training_feature_edges.shape[0])
        #     candidate_indices = farthest_first_selection.graipher(new_vectors, boost_number)
        #     new_vectors = new_vectors[candidate_indices]
        #     new_classes = new_classes[candidate_indices]
        #     agreement = new_classes == debug_labels[candidate_indices]
        #     debug_labels = debug_labels[candidate_indices]
        print("class agreement {} of {} unlabeled samples out of {}".format(sum(agreement) / agreement.shape[0],
                                                                            new_vectors.shape[0],
                                                                            unlabelled_feature_edges.shape[0]))
        print("class pos/neg {} true pos/neg ratio {}".format(sum(new_classes) / new_classes.shape[0],
                                                              sum(debug_labels) / debug_labels.shape[0]))
    return new_vectors, new_classes
