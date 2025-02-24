from random import Random

import numpy as np
from numpy import ndarray


def select_blocking_keys(rec_dict_a, rec_dict_b, blocking_key_candidates, ground_truth_pairs, training_size,
                         eps, max_block_size_ratio):
    '''
    Determines a list of blocking keys that is used for disjunctive blocking based on a candidate list. The method
    generates a balanced training data_io set using the ground truth. The candidate list is filtered by the max_block_size_ratio.
    The Fisher score is computed for each remaining blocking key. The candidate blocking keys are sorted by the Fisher score
    in a descending order. In each iteration, a blocking key bk is added to the final result, if at least one record pair is
    covered by the current blocking key, and not by the previous ones. The loop terminates, if the number of uncovered record
    pairs of M is smaller than eps * |M|
    :param rec_dict_a: record dictionary with rec id and list of value pairs for data_io data_io source A
    :param rec_dict_b: record dictionary with rec id and list of value pairs for data_io data_io source B
    :param blocking_key_candidates: list of blocking key candidates
    :param ground_truth_pairs: set of real matches
    :param training_size: number of positive training samples
    :param eps: allowed ratio of uncovered record pairs
    :param max_block_size_ratio: ratio of maximum block size regarding the total number of records
    :return: list of blocking key for a DNF blocking scheme
    '''
    positive_pairs, negative_pairs = generate_samples(rec_dict_a, rec_dict_b, ground_truth_pairs, training_size)
    number_of_uncovered_recs = eps * training_size
    max_block_size = max_block_size_ratio * max(len(rec_dict_a), len(rec_dict_b))
    print("max block size:" + str(max_block_size))
    max_block_sizes = get_max_block_size(rec_dict_a, rec_dict_b, blocking_key_candidates)
    new_filtered_candidates = []
    print(max_block_sizes)
    for index, bk in enumerate(blocking_key_candidates):
        if max_block_sizes[index] < max_block_size:
            new_filtered_candidates.append(blocking_key_candidates[index])
    pf_vectors = generate_feature_vectors(rec_dict_a, rec_dict_b, positive_pairs, new_filtered_candidates)
    nf_vectors = generate_feature_vectors(rec_dict_a, rec_dict_b, negative_pairs, new_filtered_candidates)
    fisher_scores = compute_fisher_score(pf_vectors, nf_vectors)
    print("fisher score{}".format(fisher_scores))
    candidates_score_list = list(zip([index for index in range(len(new_filtered_candidates))], fisher_scores.tolist()))
    candidates_score_list = sorted(candidates_score_list, key=lambda cand: cand[1], reverse=True)
    covered_rec_pairs = set()
    final_bks = []
    for bk_index, score in candidates_score_list:
        indices = np.where(pf_vectors[:, bk_index] == 1)
        indices = set(indices[0])
        # check if more matches are covered
        diff = indices.difference(covered_rec_pairs)
        if len(diff) > 0:
            covered_rec_pairs = covered_rec_pairs.union(diff)
            final_bks.append(new_filtered_candidates[bk_index])
        if training_size - len(covered_rec_pairs) < number_of_uncovered_recs:
            break
    print("selected DNF blocking scheme")
    print(final_bks)
    return final_bks


def get_max_block_size(rec_dict_a, rec_dict_b, blocking_key_candidates):
    max_block_size_list = []
    for bf, a in blocking_key_candidates:
        block_dict = {}
        for rec_id, rec_values in rec_dict_a.items():
            rec_bkv = bf(rec_values, a)
            if rec_bkv in block_dict:
                rec_id_list = block_dict[rec_bkv]
                rec_id_list.append(rec_id)
            else:
                rec_id_list = [rec_id]
            block_dict[rec_bkv] = rec_id_list  # Store the new block
        max_block_size = max([len(rec_list) for rec_list in block_dict.values()])
        max_block_size_list.append(max_block_size)
    att_index = 0
    for bf, a in blocking_key_candidates:
        block_dict = {}
        for rec_id, rec_values in rec_dict_b.items():
            rec_bkv = bf(rec_values, a)
            if rec_bkv in block_dict:
                rec_id_list = block_dict[rec_bkv]
                rec_id_list.append(rec_id)
            else:
                rec_id_list = [rec_id]
            block_dict[rec_bkv] = rec_id_list  # Store the new block
        max_block_size = max([len(rec_list) for rec_list in block_dict.values()])
        max_block_size_list[att_index] = max(max_block_size_list[att_index], max_block_size)
        att_index += 1
    return max_block_size_list


def generate_samples(rec_dict_a, rec_dict_b, ground_truth_pairs, training_size):
    '''
    generate a training data_io set consisting of record pairs
    :param rec_dict_a: record dictionary with rec id and list of value pairs for data_io data_io source A
    :param rec_dict_b: record dictionary with rec id and list of value pairs for data_io data_io source B
    :param ground_truth_pairs:
    :param training_size: number of true matches
    :return: set of matches, and set of non-matches
    '''
    records_A = list(rec_dict_a.keys())
    r = Random(42)
    r.shuffle(records_A)
    records_B = list(rec_dict_b.keys())
    r.shuffle(records_B)
    negative_pairs = set(zip(records_A, records_B))
    negative_pairs = set(list(negative_pairs.difference(ground_truth_pairs))[:training_size])
    positive_pairs = set(list(ground_truth_pairs)[:training_size])
    return positive_pairs, negative_pairs


# TODO Implement the generation of a set of vectors for computing the Fisher score
def generate_feature_vectors(rec_dict_a, rec_dict_b, pair_set, blocking_key_candidates):
    '''
    This method generates for a set of record pairs a set of vectors. The set is represented as numpy array. The
    i-th entry of a vector represents a record pair and is set to 1 if the blocking key values are the same
    determined by the i-th blocking key of blocking_key_candidates
    :param rec_dict_a: dictionary consisting of rec ids and values from data_io source A
    :param rec_dict_b: dictionary consisting of rec ids and values from data_io source B
    :param pair_set: set of record pairs
    :param blocking_key_candidates: list of blocking keys where each blocking key is a tuple with a blocking function
    and an attribute
    :return: numpy array with shape (number of record pairs, number of blocking keys)
    '''
    feature_vector_array = np.zeros((len(pair_set), len(blocking_key_candidates)))
    index = 0
    for r, s in pair_set:
        for index_f, bk in enumerate(blocking_key_candidates):
            rec_values_a = rec_dict_a[r]
            rec_values_b = rec_dict_b[s]
            rkv_a = bk[0](rec_values_a, bk[1])
            rkv_b = bk[0](rec_values_b, bk[1])
            if rkv_a == rkv_b:
                feature_vector_array[index][index_f] = 1
        index += 1
    return feature_vector_array


# TODO Implement the Computation of the Fisher score for blocking keys
def compute_fisher_score(pf_vectors: ndarray, nf_vectors: ndarray):
    '''
    Computes the fisher scores for all blocking key candidates. The pf_vectors and pn_vectors
    consist of vectors where each vector represents a record pair being a match (pf_vectors) or
    a non-match(pn_vectors). The i-th position of a vector representing a record pair (r,s)
    is 1 if the blocking values bf(r[a]) and bf(s[a]) regarding the i-th blocking key
    (attribute a, blocking function bf) are the same.
    :param pf_vectors: numpy array of the dimension |positive_pairs| X |blocking_key_candidates|
    :param nf_vectors: numpy array of the dimension |negative_pairs| X |blocking_key_candidates|
    :return: fisherScore
    '''

    mean_pf_i = np.mean(pf_vectors, axis=0)
    mean_nf_i = np.mean(nf_vectors, axis=0)
    var_pf_i = np.var(pf_vectors, axis=0)
    var_nf_i = np.var(nf_vectors, axis=0)
    union = np.vstack((pf_vectors, nf_vectors))
    mean_total = np.mean(union, axis=0)
    fisher_scores = pf_vectors.shape[0] * np.power(mean_pf_i - mean_total, 2) \
                    + nf_vectors.shape[0] * np.power(mean_nf_i - mean_total, 2)

    nominator = (pf_vectors.shape[0] * np.power(var_pf_i, 2) + nf_vectors.shape[0] * np.power(var_nf_i, 2))
    nominator[nominator == 0] = 1e-6
    fisher_scores = fisher_scores / nominator
    return fisher_scores
