from statistics import mean

import numpy as np
import pandas as pd

from baseline.almser import scoreaggregation


def transform_linkage_problems_to_df(linkage_problem_tuples: list[(str, str, dict[(str, str), list[float]])],
                                     cf_splitter: str, gold_links, unsupervised_links):
    tuple_list = []
    feature_columns = []
    for lp_tuple in linkage_problem_tuples:
        source = lp_tuple[0]
        target = lp_tuple[1]
        data_source_pair = str(source) + cf_splitter + str(target)
        for rec_p, sims in lp_tuple[2].items():
            # mean_sim = mean([s if s >= 0 else 0 for s in sims])
            mean_sim = mean([s for s in sims if s >= 0])
            # very bad if -1 is there -> 0.4467
            #mean_sim = mean(sims)
            if len(feature_columns) == 0:
                feature_columns = [str(index) for index in range(len(sims))]
            t_list = [source+cf_splitter+str(rec_p[0]), target+cf_splitter+str(rec_p[1]), data_source_pair]
            t_list.append(rec_p[0])
            t_list.append(rec_p[1])
            t_list.append("{}-{}".format(rec_p[0], rec_p[1]))
            t_list.extend(sims)
            t_list.append(mean_sim)
            if len(unsupervised_links) > 0:
                if tuple(sorted(rec_p)) in unsupervised_links:
                    t_list.append(True)
                else:
                    t_list.append(False)
            else:
                t_list.append(False)
                # pass
            if tuple(sorted(rec_p)) in gold_links:
                t_list.append(True)
                #t_list.append(True)
            else:
                t_list.append(False)
                #t_list.append(False)
            tuple_list.append(tuple(t_list))
    print(len(tuple_list))
    columns = ['source', 'target', 'datasource_pair', 'source_id', 'target_id', 'pair_id']
    columns.extend(feature_columns)
    columns.append('agg_score')
    columns.append('unsupervised_label')
    columns.append('label')
    data_frame = pd.DataFrame(tuple_list, columns=columns)
    #if len(unsupervised_links) == 0:
    data = data_frame[feature_columns]
    # aggregated_scores = scoreaggregation.aggregateScores_stdpredictor(data)
    aggregated_scores = data_frame['agg_score'].to_list()
    threshold = scoreaggregation.elbow_threshold(aggregated_scores)
    print(threshold)
    # data_frame['unsupervised_label'] = np.where(data_frame['agg_score'] >= threshold, True, False)
    return data_frame

