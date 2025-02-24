import gzip
import json
import re
from json import JSONDecodeError

import pandas as pd

from meta_tl.data_io.entity import Entity
from meta_tl.data_io.test_data import famer_constant, reader

url_domain_regex = r"https?:\/\/[^\/]+(?=\/|\?)"


def read_wdc(file_name, train_labels, test_data, with_reduction_save=False):
    gold_links, negative_links = read_gold_links(train_labels)
    train_entity_ids = set()
    test_entity_ids = set()
    for u, v in gold_links:
        train_entity_ids.add(u)
        train_entity_ids.add(v)
    for u, v in negative_links:
        train_entity_ids.add(u)
        train_entity_ids.add(v)

    print("train ids: {}".format(len(train_entity_ids)))

    test_links, test_negative_links = read_gold_links(test_data)
    for u, v in test_links:
        test_entity_ids.add(u.strip())
        test_entity_ids.add(v.strip())
    for u, v in test_negative_links:
        test_entity_ids.add(u.strip())
        test_entity_ids.add(v.strip())
    print("test ids: {}".format(len(test_entity_ids)))
    train_entity_dict = {}
    test_entity_dict = {}

    with gzip.open(file_name, 'r') as f:
        if with_reduction_save:
            fout = gzip.open('datasets/wdc_computer/comp_offers_english.json.gz', 'w')
        for l in f:
            try:
                property_dict = json.loads(l)
                e_id = property_dict['nodeID'] + ' ' + property_dict['url']
                source = re.findall(url_domain_regex, property_dict['url'])
                real_properties = property_dict['schema.org_properties']
                trans_p = read_json_properties(real_properties)
                e = Entity(e_id, source[0], None, trans_p)
                if e_id.strip() in train_entity_ids:
                    if with_reduction_save:
                        fout.write(l)
                    train_entity_dict[e_id.strip()] = e
                if e_id.strip() in test_entity_ids:
                    if with_reduction_save:
                        fout.write(l)
                    test_entity_dict[e_id.strip()] = e
            except IndexError:
                print(property_dict)
        print(len(test_entity_ids.intersection(set(test_entity_dict.keys()))))
        print("entities not in sources: {}".format(len(test_entity_ids.difference(set(test_entity_dict.keys())))))
        if with_reduction_save:
            fout.close()
        f.close()
    test_data_sources, test_att_indices = transform_to_data_sources(test_entity_dict)
    train_data_sources, train_att_indices = transform_to_data_sources(train_entity_dict)
    return train_data_sources, train_att_indices, test_data_sources, test_att_indices


def read_wdc_links(train_labels, test_data):
    train_tp_links, train_tn_links = read_gold_links(train_labels)
    train_entity_ids = set()
    test_entity_ids = set()
    for u, v in train_tp_links:
        train_entity_ids.add(u)
        train_entity_ids.add(v)
    for u, v in train_tn_links:
        train_entity_ids.add(u)
        train_entity_ids.add(v)
    print(len(train_entity_ids))
    tp_test_links, test_tn_links = read_gold_links(test_data)
    for u, v in tp_test_links:
        test_entity_ids.add(u)
        test_entity_ids.add(v)
    for u, v in test_tn_links:
        test_entity_ids.add(u)
        test_entity_ids.add(v)
    print(len(test_entity_ids))
    print("test links {}".format(len(tp_test_links) + len(test_tn_links)))
    return train_tp_links, train_tn_links, tp_test_links, test_tn_links


def read_json_properties(properties):
    transformed_properties = {}
    for prop_dict in properties:
        for p, atomic_v in prop_dict.items():
            str_list = atomic_v.replace('[', '').replace(']', '')
            values = str_list.split(',')
            result = [v.strip() for v in values]
            result = [re.sub(r"[^a-zA-Z0-9]+", ' ', v) for v in result]
            result = [v.strip() if v.strip() != 'null' and len(v) > 0 else '' for v in result]
            transformed_properties[p.replace('/', '')] = result[0]
    return transformed_properties


def read_gold_links(train_labels):
    tp_links = set()
    tn_links = set()
    with open(train_labels, 'r') as f:
        for line in f:
            values = line.split("#####")
            if int(values[2].strip()) == 1:
                tp_links.add(tuple(sorted([values[0].strip(), values[1].strip()])))
            else:
                tn_links.add(tuple(sorted([values[0].strip(), values[1].strip()])))
        f.close()
    return tp_links, tn_links


def transform_to_data_sources(entity_dict):
    data_sources = {}
    att_indices_dict = {}
    for id, e in entity_dict.items():
        if e.resource not in att_indices_dict:
            att_indices_dict[e.resource] = {}
        att_indices_source = att_indices_dict[e.resource]
        for att, value in e.properties.items():
            if att not in att_indices_source:
                att_indices_source[att] = len(att_indices_source)
    for id, e2 in entity_dict.items():
        if e2.resource not in data_sources:
            data_sources[e2.resource] = {}
        data_source = data_sources[e2.resource]
        att_indices_source = att_indices_dict[e2.resource]
        values = ['' for i in range(len(att_indices_source))]
        for att, value in e2.properties.items():
            values[att_indices_source[att]] = value
        data_source[id] = values
    return data_sources, att_indices_dict


if __name__ == '__main__':
    read_wdc('datasets/wdc_computer/comp_offers_english.json.gz', 'datasets/wdc_computer/wdc_train.txt',
             'datasets/wdc_computer/gs_computers.txt', False)
