'''This script splits a benchmark data set to the raw files and generate the ground truth clusters for the test set'''
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

print()
sys.path.insert(0, os.getcwd())
from meta_tl.data_io import linkage_problem_io
from meta_tl.data_io.entity import Entity
from meta_tl.data_io.test_data import reader, wdc_reader, almser_linkage_reader, famer_constant
from meta_tl.transfer.incremental.util import split_linkage_problem_tasks
from record_linkage.evaluation import metrics


def generate_raw_files(input_folder):
    pass


def main(input_folder: Annotated[
    Path,
    typer.Argument(
        exists=False,
        file_okay=True,
        dir_okay=True,
        writable=False,
        readable=True,
        resolve_path=True,
        help="Path to comparison.csv",
    ),
],
         linkage_path: Annotated[
             Path,
             typer.Argument(
                 exists=False,
                 file_okay=False,
                 dir_okay=True,
                 writable=False,
                 readable=True,
                 resolve_path=True,
                 help="Path to comparison.csv",
             ),
         ],
         result_path: Annotated[
             Path,
             typer.Argument(
                 exists=True,
                 file_okay=True,
                 dir_okay=True,
                 writable=True,
                 readable=True,
                 resolve_path=True,
                 help="Path to comparison.csv",
             ),
         ],
         ground_truth: Annotated[
             Path,
             typer.Argument(
                 exists=False,
                 file_okay=True,
                 dir_okay=True,
                 writable=True,
                 readable=False,
                 resolve_path=True,
                 help="Path to comparison.csv",
             ),
         ],
        test_path: Annotated[
             Path,
             typer.Argument(
                 exists=False,
                 file_okay=True,
                 dir_okay=True,
                 writable=True,
                 readable=False,
                 resolve_path=True,
                 help="Path to comparison.csv",
             ),
         ],
         wdc_data_set_path: Annotated[
             Path,
             typer.Option(
                 exists=True,
                 file_okay=True,
                 dir_okay=True,
                 writable=False,
                 readable=True,
                 resolve_path=True,
                 help="data set file",
             ),
         ] = None,
         train_file: Annotated[
             Path,
             typer.Option(
                 exists=True,
                 file_okay=True,
                 dir_okay=False,
                 writable=False,
                 readable=True,
                 resolve_path=True,
                 help="Output path of plot",
             ),
         ] = None,
         test_file: Annotated[
             Path,
             typer.Option(
                 exists=True,
                 file_okay=True,
                 dir_okay=False,
                 writable=False,
                 readable=True,
                 resolve_path=True,
                 help="Output path of plot",
             ),
         ] = None
         ):
    MAIN_PATH = os.getcwd()
    print(f"Read folder {input_folder}")
    if 'dexter' in str(input_folder):
        file_name = os.path.join(MAIN_PATH, input_folder)
        entities, _, _ = reader.read_data(file_name)
        gold_clusters = reader.generate_gold_clusters(entities)
        gold_links = metrics.generate_links(gold_clusters)
        data_sources_dict, data_sources_headers = reader.transform_to_data_sources(entities)
    elif 'wdc_computer' in str(input_folder):
        train_tp_links, train_tn_links, test_tp_links, test_tn_links = wdc_reader.read_wdc_links(train_file,
                                                                                                 test_file)
        gold_links = set()
        gold_links.update(train_tp_links)
        gold_links.update(test_tp_links)
    elif 'wdc_almser' in str(input_folder) or 'music_almser' in str(input_folder):
        gold_links = set()
        train_tp_links, train_tn_links, test_tp_links, test_tn_links, unsup_train_tp_links, unsup_train_tn_links = (
            almser_linkage_reader.read_wdc_links(train_file, test_file))
        gold_links.update(train_tp_links)
        gold_links.update(test_tp_links)
        print("tps overall {}".format(len(gold_links)))

    data_source_comp: dict[(str, str):[dict[(str, str):list]]] = linkage_problem_io.read_linkage_problems(
        linkage_path, deduplication=False)
    all_pairs = set([p for lp in data_source_comp.values() for p in lp.keys()])
    # ===================================================
    # Step 1: Prepare Record Linkage Tasks
    # ===================================================

    reduced_comp = linkage_problem_io.remove_empty_problems(data_source_comp)

    linkage_problems = [(k[0], k[1], lp) for k, lp in data_source_comp.items()]
    relevant_columns = [col_index for col_index in range(len(list(linkage_problems[0][2].values())[0]))]
    if 'dexter' in str(input_folder):
        solved_problems, integrated_sources, unsolved_problems = split_linkage_problem_tasks(linkage_problems,
                                                                                             split_ratio=0.5,
                                                                                             is_shuffle=True)
        test_entities = defaultdict(set)
        for s, t, lp in unsolved_problems:
            for u, v in lp.keys():
                source_entity = entities[u]
                target_entity = entities[v]
                source_entity_list = test_entities[source_entity.resource]
                target_entity_list = test_entities[target_entity.resource]
                source_entity_list.add(source_entity)
                target_entity_list.add(target_entity)
        for s, t, lp in solved_problems:
            for u, v in lp.keys():
                source_entity = entities[u]
                target_entity = entities[v]
                source_entity_list = test_entities[source_entity.resource]
                target_entity_list = test_entities[target_entity.resource]
                source_entity_list.add(source_entity)
                target_entity_list.add(target_entity)

        print(len(test_entities))
        entity_id = write_entities_to_csv(test_entities,
                                          ['<page title>','famer_model_no_list', 'famer_mpn_list', 'famer_ean_list',
                                           'famer_product_name',
                                           'famer_model_list', 'digital zoom', 'famer_opticalzoom', 'famer_width',
                                           'famer_height',
                                           'famer_weight', 'famer_resolution_from', 'famer_resolution_to'],
                                          output_path=result_path)
        with open(ground_truth, 'w') as ground_truth_file:
            train_links = 0
            for s, t, lp in solved_problems:
                for u, v in lp.keys():
                    train_links += 1
                    if (u, v) in gold_links:
                        ground_truth_file.write('{},{},True\n'.format(entity_id[u], entity_id[v]))
                    else:
                        ground_truth_file.write('{},{},False\n'.format(entity_id[u], entity_id[v]))
            ground_truth_file.close()
            print(f"train links {train_links}")
        with open(test_path, 'w') as test_file:
            for s, t, lp in unsolved_problems:
                for u, v in lp.keys():
                    if (u, v) in gold_links:
                        test_file.write('{},{},True\n'.format(entity_id[u], entity_id[v]))
                    else:
                        test_file.write('{},{},False\n'.format(entity_id[u], entity_id[v]))
        test_file.close()
    elif 'wdc_almser' in str(input_folder) or 'music_almser' in str(input_folder):
        #data_source_comp = wdc_linkage_reader.split_linkage_problems(args.train_pairs, args.test_pairs, data_source_comp)
        solved_problems = []
        unsolved_problems = []
        integrated_sources = set()
        tps_check = 0
        entities = read_wdc_entities(wdc_data_set_path, "id", "cluster_id")
        test_entities = defaultdict(set)
        with open(ground_truth, 'w') as ground_truth_file:
            for u, v in test_tp_links:
                ground_truth_file.write('{},{},1\n'.format(u, v))
            for u, v in test_tn_links:
                ground_truth_file.write('{},{},0\n'.format(u, v))
            ground_truth_file.close()
        for lp, sims in data_source_comp.items():
            if 'train' in lp[0]:
                solved_problems.append((lp[0], lp[1], sims))
                for p in sims.keys():
                    if p in gold_links:
                        tps_check += 1
                integrated_sources.add(lp[0].replace('_train', ''))
                integrated_sources.add(lp[1].replace('_train', ''))
            if 'test' in lp[0]:
                unsolved_problems.append((lp[0], lp[1], sims))
                for u, v in sims.keys():
                    source_entity = entities[u]
                    target_entity = entities[v]
                    source_entity_list = test_entities[source_entity.resource]
                    target_entity_list = test_entities[target_entity.resource]
                    source_entity_list.add(source_entity)
                    target_entity_list.add(target_entity)
                #['number','title','length','artist','album','year','language']["title","description","brand","Capacity","Manufacturer","Spindle Speed"]
                #['number','title','length','artist','album','year','language']["title","description","brand","Capacity","Manufacturer","Spindle Speed"]
        entity_id = write_entities_to_csv(test_entities,
                                          ["title","description","brand","Capacity","Manufacturer","Spindle Speed"],
                                          output_path=result_path)
        with open(ground_truth, 'w') as ground_truth_file:
            for u, v in test_tp_links:
                ground_truth_file.write('{},{},1\n'.format(entity_id[u], entity_id[v]))
            for u, v in test_tn_links:
                ground_truth_file.write('{},{},0\n'.format(entity_id[u], entity_id[v]))
            ground_truth_file.close()
        print("number of train tps in lps {}".format(tps_check))


def write_entities_to_csv(test_entities, considered_attributes, output_path):
    index = 0
    entity_id = dict()
    with open(os.path.join(output_path, f"all_sources.csv"), 'w', encoding='utf-8') as all_file:
        all_file.write("tid|id|")
        for col_index, a in enumerate(considered_attributes):
            if col_index < len(considered_attributes) - 1:
                all_file.write(a + "|")
            else:
                all_file.write(a + "\n")
        for test_entities in test_entities.values():
            with open(os.path.join(output_path, f"table_{index}.csv"), 'w', encoding='utf-8') as csvfile:
                csvfile.write("tid|id|")
                for col_index, a in enumerate(considered_attributes):
                    if col_index < len(considered_attributes) - 1:
                        csvfile.write(a + "|")
                    else:
                        csvfile.write(a + "\n")
                for e in test_entities:
                    entity_id[e.iri] = len(entity_id)
                    csvfile.write(str(entity_id[e.iri]) + "|")
                    csvfile.write("{}|".format(e.properties[famer_constant.REC_ID]))
                    all_file.write(str(entity_id[e.iri]) + "|")
                    all_file.write("{}|".format(e.properties[famer_constant.REC_ID]))
                    for col_index, a in enumerate(considered_attributes):
                        if a in e.properties:
                            if type(e.properties[a]) == str:
                                value = e.properties[a].replace('|', '\\|')
                            else:
                                value = e.properties[a]
                        else:
                            value = ''
                        if col_index < len(considered_attributes) - 1:
                            csvfile.write("{}|".format(value))
                            all_file.write("{}|".format(value))
                        else:
                            csvfile.write("{}\n".format(value))
                            all_file.write("{}\n".format(value))
                index += 1
                csvfile.close()
    return entity_id


def read_wdc_entities(input_folder: Path, entity_id_col, cluster_id_col):
    file_names = os.listdir(input_folder)
    file_names = sorted(file_names, reverse=True)
    entities = dict()
    for fn in file_names:
        path = os.path.join(input_folder, fn)
        if Path(path).is_file():
            print(path)
            data_frame = pd.read_csv(path, sep=';')
            properties = data_frame.columns
            print(properties)
            for index, row in data_frame.iterrows():
                pv_values = {}
                ent_id = row[entity_id_col]
                for p in properties:
                    if p != entity_id_col:
                        if p == cluster_id_col:
                            pv_values[famer_constant.REC_ID] = row[p]
                        else:
                            pv_values[p] = row[p]
                e = Entity(ent_id, fn, None, pv_values)
                entities[e.iri] = e
    return entities


if __name__ == '__main__':
    typer.run(main)
