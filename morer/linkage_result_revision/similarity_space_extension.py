from morer.attribute_matching.string_instance_matching import QGramInstanceMatching
from morer.feature_discovery import classification_result_analysis


class SimilaritySpaceExtension:

    def __init__(self):
        pass

    def extend_ml_models(self, data_source_dictionary, header_list_dict, result_dictionary: dict[(str, str):(set, set, dict)], is_plot=False,
                         true_match_set=set()):
        attribute_qgram = QGramInstanceMatching(qgram=2)
        for task, result in result_dictionary.items():
            data_source_a = data_source_dictionary[task[0]]
            header_list_a = header_list_dict[task[0]]
            data_source_b = data_source_dictionary[task[1]]
            header_list_b = header_list_dict[task[1]]
            a_values = list(data_source_a.values())[0]
            b_values = list(data_source_b.values())[0]
            att_a_indices = [i for i in range(1, len(a_values))]
            att_b_indices = [i for i in range(1, len(b_values))]
            attribute_pairs = attribute_qgram.generate_attribute_pairs(recA_dict, recB_dict, att_a_indices,
                                                                       att_b_indices, 0.1,
                                                                       False, is_top=True)
            att_a_indices = [i for i in range(1, len(a_values))]
            att_b_indices = [i for i in range(1, len(b_values))]
            attribute_pairs = attribute_qgram.generate_attribute_pairs(recA_dict, recB_dict, att_a_indices,
                                                                       att_b_indices, 0.1,
                                                                       False, is_top=True)
            class_match_set = result[0]
            pair_confidence = result[2]
            sorted_scores, class_match_set = classification_result_analysis.determine_false_positives(class_match_set,
                                                                                                      pair_confidence,
                                                                                                      False,
                                                                                                      is_plot=True,
                                                                                                      true_match_set=true_match_set)

