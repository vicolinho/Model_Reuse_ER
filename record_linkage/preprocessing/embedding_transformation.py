from collections import defaultdict
from itertools import chain

import numpy as np
from sentence_transformers import SentenceTransformer


class RecordTransformer:

    def __init__(self, model_path="all-MiniLM-L12-v2", device="cpu", seed=3447, max_seq_length=64, batch_size=512):
        self.model_path = model_path
        self.device = device
        self.seed = seed
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size

    def convert_to_embeddings(self, rec_dict: dict, attributes) -> dict[str:set]:
        model = SentenceTransformer(self.model_path)
        converted_rec_dict = dict()
        table_sentences = []
        attribute_sentences = defaultdict(list)
        for rec_id, values in rec_dict.items():
            sentence = ''
            for index, a in enumerate(attributes):
                pad_value = values[a]
                if type(pad_value) == str:
                    try:
                        attribute_sentences[a].append(str(pad_value))
                        if index != len(attributes) - 1:
                            sentence += str(values[a]) + ' '
                        else:
                            sentence += str(values[a])
                    except IndexError as e:
                        print(e)
            table_sentences.append(sentence)
        model.max_seq_length = self.max_seq_length
        model.to(self.device)
        table_embeddings = [
            model.encode(sentences, show_progress_bar=False,
                         batch_size=self.batch_size, normalize_embeddings=True)
            for sentences in table_sentences]
        all_embeddings = list(chain(*table_embeddings))
        all_embeddings = np.array(all_embeddings)
        attribute_embeddings_dict = dict()
        for a, att_sentences in attribute_sentences.items():
            att_embeddings = [
                model.encode(sentences, show_progress_bar=False,
                             batch_size=self.batch_size, normalize_embeddings=True)
                for sentences in att_sentences]
            att_embeddings = list(chain(*att_embeddings))
            att_embeddings = np.array(att_embeddings)
            attribute_embeddings_dict[a] = att_embeddings
        index = 0
        for rec_id, values in rec_dict.items():
            new_values = list(values)
            for a in attributes:
                new_values[a] = attribute_embeddings_dict[a][index]
            new_values.append(all_embeddings[index])
            converted_rec_dict[rec_id] = (values, new_values)
            index += 1
        return converted_rec_dict
