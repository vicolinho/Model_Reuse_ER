import math


def cluster_occurrence_scoring(data_source_comp: dict, selected_tasks: dict[(str,str):list]):
    '''
    Determine for each record a score based on the occurrence in different cluster. The intention
    is to weigh links higher with adjacent records occurring in multiple clusters.
    :param data_source_comp:
    :param selected_tasks:
    :return: dictionary of record-pairs and their scores
    '''
    cluster_inter_occurrence = {}
    cluster_intra_occurrence = {}
    for cluster_task, task_list in selected_tasks.items():
        cluster_intra_occurrence[cluster_task] = {}
        for task in task_list:
            lp_problem = data_source_comp[task[0]]
            for p in lp_problem.keys():
                if p[0] not in cluster_inter_occurrence:
                    cluster_inter_occurrence[p[0]] = set()
                if p[1] not in cluster_inter_occurrence:
                    cluster_inter_occurrence[p[1]] = set()
                cluster_inter_occurrence[p[0]].add(cluster_task)
                cluster_inter_occurrence[p[1]].add(cluster_task)
                if p[0] not in cluster_intra_occurrence[cluster_task]:
                    cluster_intra_occurrence[cluster_task][p[0]] = set()
                if p[1] not in cluster_intra_occurrence[cluster_task]:
                    cluster_intra_occurrence[cluster_task][p[1]] = set()
                cluster_intra_occurrence[cluster_task][p[0]].add(task[0])
                cluster_intra_occurrence[cluster_task][p[1]].add(task[0])
    inter_cluster_occurrence_scores = {}
    intra_cluster_occurrence_scores = {}
    # assume that record pairs with adjacent records occuring in multiple clusters are relevant
    for record, tasks in cluster_inter_occurrence.items():
        inter_cluster_occurrence_scores[record] = math.log(len(selected_tasks)/len(tasks))
    for cluster_task, intra_tasks_dict in cluster_intra_occurrence.items():
        intra_cluster_occurrence_scores[cluster_task] = {}
        for r, tasks in intra_tasks_dict.items():
            # intra_cluster_occurrence_scores[cluster_task][r] = float(len(tasks))/len(selected_tasks[cluster_task])
            intra_cluster_occurrence_scores[cluster_task][r] = float(len(tasks)) * inter_cluster_occurrence_scores[r]
    return inter_cluster_occurrence_scores, intra_cluster_occurrence_scores


