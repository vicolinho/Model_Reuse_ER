


def count_total_number_of_links(data_source_comp:dict[(str, str):[dict[(str, str):list]]]):
    total_pairs = 0
    for p, sim_vecs in data_source_comp.items():
        total_pairs += len(sim_vecs)
    return total_pairs
