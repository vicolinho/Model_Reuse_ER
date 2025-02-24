""" Module with functionalities for classifying a dictionary of record pairs
    and their similarities based on a similarity threshold.

    Each function in this module returns two sets, one with record pairs
    classified as matches and the other with record pairs classified as
    non-matches.
"""


# =============================================================================
import math


def exact_classify(sim_vec_dict):
    """Method to classify the given similarity vector dictionary assuming only
     exact matches (having all similarities of 1.0) are matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.

     The classification is based on the exact matching of attribute values,
     that is the similarity vector for a given record pair must contain 1.0
     for all attribute values.

     Example:
       (recA1, recB1) = [1.0, 1.0, 1.0, 1.0] => match
       (recA2, recB5) = [0.0, 1.0, 0.0, 1.0] = non-match
  """

    print('Exact classification of %d record pairs' % (len(sim_vec_dict)))

    class_match_set = set()
    class_nonmatch_set = set()

    # Iterate over all record pairs
    #
    for (rec_id_tuple, sim_vec) in sim_vec_dict.items():

        sim_sum = sum(sim_vec)  # Sum all attribute similarities

        if sim_sum == len(sim_vec):  # All similarities were 1.0
            class_match_set.add(rec_id_tuple)
        else:
            class_nonmatch_set.add(rec_id_tuple)

    print('  Classified %d record pairs as matches and %d as non-matches' % \
          (len(class_match_set), len(class_nonmatch_set)))
    print('')

    return class_match_set, class_nonmatch_set


# -----------------------------------------------------------------------------
# TODO Implement the threshold classification
def threshold_classify(sim_vec_dict, sim_thres):
    """Method to classify the given similarity vector dictionary with regard to
     a given similarity threshold (in the range 0.0 to 1.0), where record pairs
     with an average similarity of at least this threshold are classified as
     matches and all others as non-matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.
       sim_thres    : The classification similarity threshold.
  """

    assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

    print('Similarity threshold based classification of %d record pairs' % \
          (len(sim_vec_dict)))
    print('  Classification similarity threshold: %.3f' % (sim_thres))

    class_match_set = set()
    class_nonmatch_set = set()

    # Iterate over all record pairs
    #
    pair_confidence = {}
    for (rec_id_tuple, sim_vec) in sim_vec_dict.items():
        sim_sum = float(sum(sim_vec))  # Sum all attribute similarities
        avr_sim = sim_sum / len(sim_vec)

        if avr_sim >= sim_thres:  # Average similarity is high enough
            class_match_set.add(rec_id_tuple)
            pair_confidence[rec_id_tuple] = avr_sim
        else:
            class_nonmatch_set.add(rec_id_tuple)

        # ************ End of your code *******************************************

    print('  Classified %d record pairs as matches and %d as non-matches' % \
          (len(class_match_set), len(class_nonmatch_set)))
    print('')

    return class_match_set, class_nonmatch_set, pair_confidence


# -----------------------------------------------------------------------------
# TODO Implement the minimum threshold classification
def min_threshold_classify(sim_vec_dict, sim_thres):
    """Method to classify the given similarity vector dictionary with regard to
     a given similarity threshold (in the range 0.0 to 1.0), where record pairs
     that have all their similarities (of all attributes compared) with at
     least this threshold are classified as matches and all others as
     non-matches.

     Parameter Description:
       sim_vec_dict : Dictionary of record pairs with their identifiers as
                      as keys and their corresponding similarity vectors as
                      values.
       sim_thres    : The classification minimum similarity threshold.
  """

    assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

    print('Minimum similarity threshold based classification of ' + \
          '%d record pairs' % (len(sim_vec_dict)))
    print('  Classification similarity threshold: %.3f' % (sim_thres))

    class_match_set = set()
    class_nonmatch_set = set()

    # Iterate over all record pairs
    #
    pair_confidence = {}
    for (rec_id_tuple, sim_vec) in sim_vec_dict.items():
        record_pair_match = True

        # check for all the compared attributes
        #
        min_sim = min(sim_vec)
        for sim in sim_vec:
            if sim < sim_thres:  # Similarity is not enough
                record_pair_match = False
                break  # No need to compare more similarities, speed-up the process

        if record_pair_match:  # All similaries are high enough
            class_match_set.add(rec_id_tuple)
            pair_confidence[rec_id_tuple] = min_sim
        else:
            class_nonmatch_set.add(rec_id_tuple)

    print('  Classified %d record pairs as matches and %d as non-matches' % \
          (len(class_match_set), len(class_nonmatch_set)))
    print('')

    return class_match_set, class_nonmatch_set, pair_confidence


# -----------------------------------------------------------------------------

def weighted_similarity_classify(sim_vec_dict, weight_vec, sim_thres):
    """Method to classify the given similarity vector dictionary with regard to
   a given weight vector and a given similarity threshold (in the range 0.0
   to 1.0), where an overall similarity is calculated based on the weights
   for each attribute, and where record pairs with the similarity of at least
   the given threshold are classified as matches and all others as
   non-matches.

   Parameter Description:
     sim_vec_dict : Dictionary of record pairs with their identifiers as
                    as keys and their corresponding similarity vectors as
                    values.
     weight_vec   : A vector with weights, one weight for each attribute.
     sim_thres    : The classification similarity threshold.
"""

    assert sim_thres >= 0.0 and sim_thres <= 1.0, sim_thres

    # Check weights are available for all attributes
    #
    first_sim_vec = list(sim_vec_dict.values())[0]
    assert len(weight_vec) == len(first_sim_vec), "sim vector{} weight{}".format(len(first_sim_vec), len(weight_vec))

    print('Weighted similarity based classification of %d record pairs' % \
          (len(sim_vec_dict)))
    print('  Weight vector: %s' % (str(weight_vec)))
    print('  Classification similarity threshold: %.3f' % (sim_thres))

    class_match_set = set()
    class_nonmatch_set = set()

    weight_sum = sum(weight_vec)  # Sum of all attribute weights

    # Iterate over all record pairs
    #
    pair_confidence = {}
    for (rec_id_tuple, sim_vec) in sim_vec_dict.items():
        # ******* Implement weighted similarity classification ********************
        sim = 0
        for i in range(len(sim_vec)):
            sim += sim_vec[i] * weight_vec[i] / weight_sum
        if sim >= sim_thres:
            class_match_set.add(rec_id_tuple)
            pair_confidence[rec_id_tuple] = sim
        else:
            class_nonmatch_set.add(rec_id_tuple)

        # ************ End of your code *******************************************

    print('  Classified %d record pairs as matches and %d as non-matches' % \
          (len(class_match_set), len(class_nonmatch_set)))
    print('')

    return class_match_set, class_nonmatch_set, pair_confidence


def automatic_weight_computation(rec_dict_a: dict, rec_dict_b: dict, compared_attribute_idx):
    unique_att_values = dict()
    for aid in compared_attribute_idx:
        unique_att_values[aid] = set()
    if rec_dict_a is not None:
        for rec, values in rec_dict_a.items():
            for aid in compared_attribute_idx:
                att_values = unique_att_values[aid]
                att_values.add(str(values[aid]))
    if rec_dict_b is not None:
        for rec, values in rec_dict_b.items():
            for aid in compared_attribute_idx:
                att_values = unique_att_values[aid]
                att_values.add(str(values[aid]))
    weight_vector = []
    weight_vector_dict = {}
    sum = 0
    for aid in compared_attribute_idx:
        weight_vector.append(math.log2(len(unique_att_values[aid])))
        weight_vector_dict[aid] = math.log2(len(unique_att_values[aid]))
        sum += math.log2(len(unique_att_values[aid]))
    # for i in range(len(weight_vector)):
    #    weight_vector[i] /= sum
    return weight_vector, weight_vector_dict
