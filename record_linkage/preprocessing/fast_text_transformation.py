import fasttext
import numpy as np


class FastTextTransformer:


    def __init__(self, fasttext_file):
        self.model = fasttext.load_model(fasttext_file)

    def convert_to_fasttext_embeddings(self,rec_dict: dict, attributes) -> dict[str:set]:
        converted_rec_dict = dict()
        for rec_id, values in rec_dict.items():
            new_values = list(values)
            for a in attributes:
                pad_value = values[a]
                q_gram_set = set(pad_value.split())
                value_array = []
                for w in q_gram_set:
                    array = self.model.get_word_vector(w.lower())
                    array[np.isnan(array)] = 0
                    if not np.isnan(array).any():
                        value_array.append(array)
                    else:
                        print(w)
                embedding = np.mean(np.asarray(value_array), axis=0)
                new_values[a] = embedding
            converted_rec_dict[rec_id] = (values, new_values)
        return converted_rec_dict