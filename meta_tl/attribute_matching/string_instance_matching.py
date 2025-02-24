from meta_tl.attribute_matching.attribute_instance_matcher import AttributeInstanceMatcher
from record_linkage.comparison.qgram_converter import QgramConverter
from collections import Counter


class QGramInstanceMatching(AttributeInstanceMatcher):

    def __init__(self, qgram=2):
        self.q_gram = qgram

    def generate_attribute_pairs(self, rec_A_dict: dict[str:list], rec_B_dict: dict[str:list],
                                 att_a_indices, att_b_indices, threshold=0.2, is_qgram=False, is_top=True):
        attribute_pairs = []
        converter = QgramConverter()
        print(len(att_a_indices))
        if is_qgram:
            convertA = converter.convert_to_qgrams(rec_A_dict, att_a_indices, True, self.q_gram)
            convertB = converter.convert_to_qgrams(rec_B_dict, att_b_indices, True, self.q_gram)
        else:
            convertA = converter.convert_to_words(rec_A_dict, att_a_indices)
            convertB = converter.convert_to_words(rec_B_dict, att_b_indices)
        att_a_values = self.convert_records_to_attribute_values(convertA, att_a_indices)
        att_b_values = self.convert_records_to_attribute_values(convertB, att_b_indices)
        counter_a_dict = {}
        counter_b_dict = {}
        for a, values in att_a_values.items():
            a_counter = Counter(values)
            if '' in a_counter:
                a_counter.pop('')
            if '#'*(self.q_gram*2) in a_counter:
                a_counter.pop('#'*(self.q_gram*2))
            counter_a_dict[a] = a_counter
        for a, values in att_b_values.items():
            b_counter = Counter(values)
            if '' in b_counter:
                b_counter.pop('')
            if '#' * (self.q_gram * 2) in b_counter:
                b_counter.pop('#' * (self.q_gram * 2))
            counter_b_dict[a] = b_counter
        print(counter_a_dict.keys())
        print(counter_b_dict.keys())

        if len(att_a_indices) < len(att_b_indices):
            min_att_indices = att_a_indices
            other_list = att_b_indices
            is_top_a = True
        else:
            min_att_indices = att_b_indices
            other_list = att_a_indices
            is_top_a = False

        for a_index in min_att_indices:
            max_intersection = 0
            current_index = -1
            for b_index in other_list:
                histo_sum = 0
                unique_qgrams = set()
                counter_a = counter_a_dict[a_index]
                counter_b = counter_b_dict[b_index]
                for q_gram, c in counter_a.items():
                    unique_qgrams.add(q_gram)
                    if q_gram in counter_b:
                        histo_sum += min(c, counter_b[q_gram])
                for q_gram, c in counter_b.items():
                    unique_qgrams.add(q_gram)
                if len(unique_qgrams) > 0:
                    histogram_intersection = histo_sum/len(unique_qgrams)/(len(rec_A_dict)+len(rec_B_dict))
                if histogram_intersection > max_intersection:
                    max_intersection = histogram_intersection
                    current_index = b_index
                print("{}-{}:{}".format(a_index, b_index, histogram_intersection))
                if not is_top:
                    if histogram_intersection > threshold:
                        attribute_pairs.append((a_index, b_index, histogram_intersection))
            if current_index != -1 and is_top:
                if is_top_a:
                    attribute_pairs.append((a_index, current_index, max_intersection))
                else:
                    attribute_pairs.append((current_index, a_index, max_intersection))
        return attribute_pairs

    def convert_records_to_attribute_values(self, rec_dict: dict[str:list], att_indices):
        att_source_values = {}
        for t in rec_dict.values():
            q_gram_sets = t[1]
            for a in att_indices:
                if a not in att_source_values:
                    values = []
                    att_source_values[a] = values
                values = att_source_values[a]
                values.extend(q_gram_sets[a])
        return att_source_values
