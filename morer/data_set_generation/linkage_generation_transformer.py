import argparse
import os

import numpy as np

import record_linkage.blocking.blocking_functions_solution as blocking_function
from morer.data_io.test_data import reader
from record_linkage.blocking import blocking
from record_linkage.comparison import string_functions_solution, comparison
from record_linkage.comparison.embedding_comparison import cosine_comp
from record_linkage.comparison.numerical_distances import percentage_distance
from record_linkage.comparison.qgram_converter import QgramConverter
from record_linkage.comparison.string_functions_solution import dice_comp
from morer.data_io import linkage_problem_io
from record_linkage.preprocessing.embedding_transformation import RecordTransformer
from record_linkage.utils import knn_search


def search_ij(embeddings_i: np.array, embeddings_j: np.array, k: int, seed: int, min_dis: float):
    ids_i = list(range(embeddings_i.shape[0]))
    ids_j = list(range(embeddings_j.shape[0]))
    I1, D1 = knn_search(embeddings_j, np.array(
        ids_j), embeddings_i, k, seed)
    pairs_ij = [(p, vi) for p, v, d in zip(ids_i, I1, D1)
                for vi, di in zip(v, d) if di <= min_dis]
    return pairs_ij

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rl generation')
    parser.add_argument('--data_file', '-d', type=str, default='datasets/dexter/DS-C0/SW_0.3', help='data file')
    parser.add_argument('--save_dir', '-o', type=str, default='data/linkage_problems/dexter',
                        help='linkage problem directory')
    args = parser.parse_args()
    wd = os.getcwd()
    folder = os.path.join(wd, args.save_dir)
    print(folder)
    file_name = os.path.join(wd, args.data_file)
    entities, _, _ = reader.read_data(file_name)
    data_sources_dict, data_sources_headers = reader.transform_to_data_sources(entities)
    sum_records = 0
    data_source_list = []
    for name, ds in data_sources_dict.items():
        print(len(ds))
        data_source_list.append((name, ds))
        sum_records += len(ds)
    print(sum_records)
    embedding_transformer = RecordTransformer()

    base_comparisons = [
        (dice_comp, 'famer_model_no_list', 'famer_model_no_list'),  # Modell-liste
        (dice_comp, 'famer_mpn_list', 'famer_mpn_list'),  # MPN-Liste
        (dice_comp, 'famer_ean_list', 'famer_ean_list'),  # EAN-Liste
        (dice_comp, 'famer_product_name', 'famer_product_name'),  # product-name,
        (dice_comp, 'famer_model_list', 'famer_model_list'),
        (dice_comp, 'digital zoom', 'digital zoom'),  # digital-zoom
        (percentage_distance, 'famer_opticalzoom', 'famer_opticalzoom'),  # optical-zoom
        (percentage_distance, 'famer_width', 'famer_width'),  # Breite
        (percentage_distance, 'famer_height', 'famer_height'),  # Hohe
        (percentage_distance, 'famer_weight', 'famer_weight'),  # Gewicht
        (percentage_distance, 'famer_resolution_from', 'famer_resolution_from'),
        (percentage_distance, 'famer_resolution_to', 'famer_resolution_to')]
    emb_comparisons = [
        (cosine_comp, 'famer_model_no_list', 'famer_model_no_list'),  # Modell-liste
        (cosine_comp, 'famer_mpn_list', 'famer_mpn_list'),  # MPN-Liste
        (cosine_comp, 'famer_ean_list', 'famer_ean_list'),  # EAN-Liste
        (cosine_comp, 'famer_product_name', 'famer_product_name'),  # product-name,
        (cosine_comp, 'famer_model_list', 'famer_model_list'),
        (cosine_comp, 'digital zoom', 'digital zoom'),  # digital-zoom
        (cosine_comp, 'all', 'all')
    ]
    blocking_functions = [(blocking_function.simple_blocking_key, 'famer_keys'),
                          ]
    data_source_comp = {}
    preprocessed_dict = {}
    embedding_preprocessed_dict = {}
    converter = QgramConverter()
    string_functions_solution.is_efficient = True
    values = []
    for i in range(len(data_source_list)):
        data_source_a = data_source_list[i][1]
        headers_a = data_sources_headers[data_source_list[i][0]]
        att_idx = [m for m in range(len(headers_a))]
        if data_source_list[i][0] not in preprocessed_dict:
            preprocessed_dict[data_source_list[i][0]] = converter.convert_to_qgrams(data_source_a, att_idx,
                                                                                    True, 2)
            embedding_preprocessed_dict[data_source_list[i][0]] = embedding_transformer.convert_to_embeddings(data_source_a, att_idx)
        blocking_functions_index_a = [(b[0], headers_a[b[1]]) for b in blocking_functions]
        blocks_a = blocking.conjunctive_block(data_source_a, blocking_functions_index_a)
        for k in range(i, len(data_source_list)):
            data_source_b = data_source_list[k][1]
            headers_b = data_sources_headers[data_source_list[k][0]]
            att_idx = [l for l in range(len(headers_b))]
            if data_source_list[k][0] not in preprocessed_dict:
                preprocessed_dict[data_source_list[k][0]] = converter.convert_to_qgrams(data_source_b, att_idx,
                                                                                        True, 2)
                embedding_preprocessed_dict[data_source_list[k][0]] = embedding_transformer.convert_to_embeddings(data_source_b, att_idx)
            base_comparisons_index = []
            for t in base_comparisons:
                index_a = 1000
                index_b = 1000
                if t[1] in headers_a:
                    index_a = headers_a[t[1]]
                if t[2] in headers_b:
                    index_b = headers_b[t[2]]
                base_comparisons_index.append((t[0], index_a, index_b))
            emb_comparisons_index = []
            for t in emb_comparisons:
                index_a = 1000
                index_b = 1000
                if t[1] in headers_a:
                    index_a = headers_a[t[1]]
                elif t[1] == 'all':
                    index_a = len(headers_a)
                if t[2] in headers_b:
                    index_b = headers_b[t[2]]
                elif t[2] == 'all':
                    index_b = len(headers_b)
                emb_comparisons_index.append((t[0], index_a, index_b))
            blocking_functions_index_b = [(b[0], headers_b[b[1]]) for b in blocking_functions]
            blocks_b = blocking.conjunctive_block(data_source_b, blocking_functions_index_b)
            # sim_vect = comparison.compare_blocks(blocks_a, blocks_b, preprocessed_dict[data_source_list[i][0]],
            #                                      preprocessed_dict[data_source_list[k][0]], base_comparisons_index)
            embeds = comparison.compare_blocks(blocks_a, blocks_b, embedding_preprocessed_dict[data_source_list[i][0]],
                                               embedding_preprocessed_dict[data_source_list[k][0]], emb_comparisons_index)
            data_source_comp[(data_source_list[i][0], data_source_list[k][0])] = embeds
    linkage_problem_io.dump_linkage_problems(data_source_comp, folder)
