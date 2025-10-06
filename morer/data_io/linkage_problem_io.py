import os.path
import pickle


def dump_linkage_problems(data_source_comp: dict[(str, str):[dict[(str, str):list]]], folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, 'linkage_problems_embeds.pkl')
    with open(path, 'wb') as f:
        pickle.dump(data_source_comp, f)
        f.close()


def read_linkage_problems(folder, deduplication=False) -> dict[(str, str):[dict[(str, str):list]]]:
    path = os.path.join(folder, 'linkage_problems.pkl')
    with open(path, 'rb') as f:
        data_source_comp = pickle.load(f)
        f.close()
        if deduplication:
            deduplication_source_comp = {}
            for k, v in data_source_comp.items():
                if k[0] != k[1]:
                    deduplication_source_comp[k] = v
            return deduplication_source_comp
        else:
            return data_source_comp


def remove_empty_problems(data_source_comp: dict[(str, str):[dict[(str, str):list]]]):
    pairs = list(data_source_comp.keys())
    count = len(data_source_comp)
    for pair in pairs:
        if len(data_source_comp[pair]) == 0:
            del data_source_comp[pair]
    print('removed %d data source pairs from %d', count - len(data_source_comp), count)
    return data_source_comp
