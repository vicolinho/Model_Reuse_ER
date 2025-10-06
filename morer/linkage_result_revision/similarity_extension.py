from record_linkage.comparison import comparison


def compute_additional_similarities(rec_dict_a: dict[str:tuple], rec_dict_b: dict[str:tuple],
                                    fp_candidates: list[(tuple, int)], sim_functions: list):
    extended_pairs_similarities = {}
    new_sims = []
    for p in fp_candidates:
        new_w = comparison.compare_record(rec_dict_a[p[0]], rec_dict_b[p[1]], sim_functions)
        extended_pairs_similarities[p] = new_w
        new_sims.append(new_w)
    return extended_pairs_similarities, new_sims
