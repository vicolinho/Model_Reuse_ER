import argparse
import os
import sys
sys.path.append(os.getcwd())
import record_linkage.blocking.blocking_functions_solution as blocking_function
from meta_tl.data_io.test_data import reader
from record_linkage.blocking import blocking
from record_linkage.comparison import string_functions_solution, comparison
from record_linkage.comparison.numerical_distances import percentage_distance
from record_linkage.comparison.qgram_converter import QgramConverter
from record_linkage.comparison.string_functions_solution import dice_comp
from meta_tl.data_io import linkage_problem_io

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
    blocking_functions = [(blocking_function.simple_blocking_key, 'famer_keys'),
                          ]
    data_source_comp = {}
    preprocessed_dict = {}
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
        blocking_functions_index_a = [(b[0], headers_a[b[1]]) for b in blocking_functions]
        blocks_a = blocking.conjunctive_block(data_source_a, blocking_functions_index_a)
        for k in range(i, len(data_source_list)):
            data_source_b = data_source_list[k][1]
            headers_b = data_sources_headers[data_source_list[k][0]]
            att_idx = [l for l in range(len(headers_b))]
            if data_source_list[k][0] not in preprocessed_dict:
                preprocessed_dict[data_source_list[k][0]] = converter.convert_to_qgrams(data_source_b, att_idx,
                                                                                        True, 2)
            base_comparisons_index = []
            for t in base_comparisons:
                index_a = 1000
                index_b = 1000
                if t[1] in headers_a:
                    index_a = headers_a[t[1]]
                if t[2] in headers_b:
                    index_b = headers_b[t[2]]
                base_comparisons_index.append((t[0], index_a, index_b))
            blocking_functions_index_b = [(b[0], headers_b[b[1]]) for b in blocking_functions]
            blocks_b = blocking.conjunctive_block(data_source_b, blocking_functions_index_b)
            sim_vect = comparison.compare_blocks(blocks_a, blocks_b, preprocessed_dict[data_source_list[i][0]],
                                                 preprocessed_dict[data_source_list[k][0]], base_comparisons_index)

            data_source_comp[(data_source_list[i][0], data_source_list[k][0])] = sim_vect
   # linkage_problem_io.dump_linkage_problems(data_source_comp, folder)
