import argparse
import os

from meta_tl.data_io import linkage_problem_io
from meta_tl.data_io.test_data import wdc_reader
from record_linkage.comparison import string_functions_solution, comparison, word_token_functions
from record_linkage.preprocessing.tfidf_transformation import TFIDFTransformation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rl generation')
    parser.add_argument('--data_file', '-d', type=str,
                        default='datasets/wdc_computer/comp_offers_english.json.gz', help='data file')
    parser.add_argument('--train_labels', '-l', type=str,
                        default='datasets/wdc_computer/wdc_train.txt', help='data file')
    parser.add_argument('--test_labels', '-gs', type=str,
                        default='datasets/wdc_computer/gs_computers.txt', help='data file')
    parser.add_argument('--fasttext', '-f', type=str, default='datasets/wiki.en/wiki.en.bin', help='data file')
    parser.add_argument('--save_dir', '-o', type=str, default='data/linkage_problems/wdc_computer_tfidf',
                        help='linkage problem directory')
    args = parser.parse_args()
    wd = os.getcwd()

    folder = os.path.join(wd, args.save_dir)
    print(folder)
    file_name = os.path.join(wd, args.data_file)
    (data_sources_dict, data_sources_headers,
     test_data_sources, test_data_source_headers) = wdc_reader.read_wdc(args.data_file, args.train_labels,
                                                                        args.test_labels)
    train_links, train_negative_links, test_links, test_negative_links = wdc_reader.read_wdc_links(args.train_labels,
                                                                                                   args.test_labels)
    sum_records = 0
    data_source_list = []
    for name, ds in data_sources_dict.items():
        data_source_list.append((name, ds))
        sum_records += len(ds)
    for name, ds in test_data_sources.items():
        data_source_list.append((name, ds))
        sum_records += len(ds)
    print(sum_records)
    data_sources_headers.update(test_data_source_headers)
    # base_comparisons = [
    #     (embedding_comparison.cosine_comp, 'name', 'name'),  # name
    #     (embedding_comparison.cosine_comp, 'brand', 'brand'),  # brand
    #     (embedding_comparison.cosine_comp, 'description', 'description'),  # description
    #     (embedding_comparison.cosine_comp, 'priceCurrency', 'priceCurrency'),  # price currency
    #     (embedding_comparison.cosine_comp, 'price', 'price')
    #     ]
    base_comparisons = [
        (word_token_functions.cosine_tfidf_similarity, 'name', 'name'),  # name
        (word_token_functions.cosine_tfidf_similarity, 'brand', 'brand'),  # brand
        (word_token_functions.cosine_tfidf_similarity, 'description', 'description'),  # description
        (word_token_functions.cosine_tfidf_similarity, 'priceCurrency', 'priceCurrency'),  # price currency
        (word_token_functions.cosine_tfidf_similarity, 'price', 'price')
    ]

    data_source_comp = {}
    preprocessed_dict = {}
    # converter = FastTextTransformer(args.fasttext)
    tfidf_converter = TFIDFTransformation()
    string_functions_solution.is_efficient = True
    values = []
    all_pairs = set()
    all_pairs.update(train_links)
    all_pairs.update(train_negative_links)
    all_pairs.update(test_links)
    all_pairs.update(test_negative_links)
    print(len(all_pairs))
    idf_map = tfidf_converter.generate_tfidf_matrix(data_sources_dict, data_sources_headers)
    word_token_functions.idf_map = idf_map
    for i in range(len(data_source_list)):
        data_source_a = data_source_list[i][1]
        headers_a = data_sources_headers[data_source_list[i][0]]
        att_idx = [m for m in range(len(headers_a))]
        if data_source_list[i][0] not in preprocessed_dict:
            # preprocessed_dict[data_source_list[i][0]] = converter.convert_to_fasttext_embeddings(data_source_a, att_idx)
            preprocessed_dict[data_source_list[i][0]] = tfidf_converter.convert_to_tf_maps(data_source_a, att_idx)
        for k in range(i, len(data_source_list)):
            data_source_b = data_source_list[k][1]
            headers_b = data_sources_headers[data_source_list[k][0]]
            att_idx = [l for l in range(len(headers_b))]
            if data_source_list[k][0] not in preprocessed_dict:
                preprocessed_dict[data_source_list[k][0]] = tfidf_converter.convert_to_tf_maps(data_source_b, att_idx)
            base_comparisons_index = []
            for t in base_comparisons:
                index_a = 1000
                index_b = 1000
                if t[1] in headers_a:
                    index_a = headers_a[t[1]]
                if t[2] in headers_b:
                    index_b = headers_b[t[2]]
                base_comparisons_index.append((t[0], index_a, index_b))
            sim_vect = comparison.compare_pairs(all_pairs, preprocessed_dict[data_source_list[i][0]],
                                                preprocessed_dict[data_source_list[k][0]], base_comparisons_index)

            data_source_comp[(data_source_list[i][0], data_source_list[k][0])] = sim_vect
    all_comps = set()
    for lp in data_source_comp.values():
        all_comps.update(lp.keys())
    for l in test_links.difference(all_comps):
        print(l)
    linkage_problem_io.dump_linkage_problems(data_source_comp, folder)
