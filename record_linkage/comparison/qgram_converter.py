class QgramConverter:

    def __init__(self):
        self.q_gram_dict = dict()

    def convert_to_qgrams(self, rec_dict: dict, attributes, padding, qgram)-> dict[str:set]:
        '''

        :param rec_dict: dictionary with the record id as key and a list of attribute values
        :param attributes: list of attributes that are converted to q-grams
        :param padding: boolean if padding is used or not
        :param qgram: number of characters for q-gram
        :return: overwrite the values of attributes
        '''
        converted_rec_dict = dict()
        for rec_id, values in rec_dict.items():
            new_values = list(values)
            q_gram_values = [set() for i in range(len(values))]
            for a in attributes:
                pad_value = values[a]
                if type(pad_value) == str:
                    if padding:
                        pad_value = "# " * (qgram - 1) + pad_value + "#" * (qgram - 1)
                    q_gram_set = set()

                    for i in range(len(pad_value) - (qgram - 1)):
                        qgram_value = pad_value[i:i + qgram]
                        if qgram_value not in self.q_gram_dict:
                            self.q_gram_dict[qgram_value] = len(self.q_gram_dict)
                        qid = self.q_gram_dict[qgram_value]
                        q_gram_set.add(qid)
                    #q_gram_set = set([pad_value[i:i + qgram] for i in range(len(pad_value) - (qgram - 1))])
                    q_gram_values[a] = q_gram_set
                    # new_values[a] = q_gram_set
            converted_rec_dict[rec_id] = (values, q_gram_values)
        return converted_rec_dict

    def convert_to_words(self,rec_dict: dict, attributes) -> dict[str:set]:
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
                q_gram_set = set(pad_value.split('\\s'))
                new_values[a] = q_gram_set
            converted_rec_dict[rec_id] = (values, new_values)
        return converted_rec_dict
