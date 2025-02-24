import math
from collections import Counter

import nltk
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

class TFIDFTransformation:
    def __init__(self):
        pass
    def generate_tfidf_matrix(self, data_sources:dict[str:list[str]], data_sources_headers):

        record_counter_words = {}
        number_of_records = 0
        for data_source, records in data_sources.items():
            headers_a = data_sources_headers[data_source]
            att_idx = [m for m in range(len(headers_a))]
            number_of_records += len(records)
            for rec_id, values in records.items():
                new_values = set()
                for a in att_idx:
                    pad_value = values[a]
                    q_gram_set = set(word_tokenize(pad_value))
                    new_values.update(q_gram_set)
                for s in new_values:
                    if s not in record_counter_words:
                        record_counter_words[s] = 1
                    else:
                        record_counter_words[s] += 1
        for s in record_counter_words.keys():
            record_counter_words[s] = math.log(number_of_records/record_counter_words[s])
        return record_counter_words


    def convert_to_tf_maps(self,rec_dict: dict, attributes) -> dict[str:set]:
        converted_rec_dict = dict()

        for rec_id, values in rec_dict.items():
            new_values = list(values)
            for a in attributes:
                pad_value = values[a]
                q_gram_set = set()
                # for i in range(len(pad_value) - (qgram - 1)):
                #     qgram_value = pad_value[i:i + qgram]
                #     if qgram_value not in self.q_gram_dict:
                #         self.q_gram_dict[qgram_value] = len(self.q_gram_dict)
                #     qid = self.q_gram_dict[qgram_value]
                #     q_gram_set.add(qid)
                words = word_tokenize(pad_value)
                word_count = Counter(words)
                new_values[a] = dict(word_count)
            converted_rec_dict[rec_id] = (values, new_values)
        return converted_rec_dict

