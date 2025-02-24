import os
# Define the main path for the project
#MAIN_PATH = '/home/dbs-experiments/PycharmProjects/metadatatransferlearning'
import sys
sys.path.append(os.getcwd())
MAIN_PATH = os.getcwd()
fv_splitter = "_"
# Add the path to the custom module directory

# Define the configuration parameters
ACTIVE_LEARNING_ITERATION_BUDGET = 5

data_file = 'datasets/dexter/DS-C0/SW_0.3'
# linkage_tasks_dir = 'data/linkage_problems/dexter'
# linkage_tasks_dir = 'data/linkage_problems/wdc_almser'
linkage_tasks_dir = 'data/linkage_problems/music_almser'
train_pairs = 'data/linkage_problems/music_almser/train_pairs_fv.csv'
test_pairs = 'data/linkage_problems/music_almser/test_pairs_fv.csv'
output_path = 'results'
# Active Learning Settings
max_queries = 1000
runs = 3
query_strategy = 'almser_gb' #almser_gb, uncertainty, disagreeement, almser_group, random


from meta_tl.transfer.incremental.util import split_linkage_problem_tasks
from meta_tl.data_io import linkage_problem_io
from record_linkage.classification.machine_learning import constants
from record_linkage.evaluation import metrics
from meta_tl.data_io.test_data import reader, wdc_reader, almser_linkage_reader
import os
file_name = os.path.join(MAIN_PATH, data_file)
RECORD_LINKAGE_TASKS_PATH = os.path.join(MAIN_PATH, linkage_tasks_dir)
ML_MODEL = constants.RF
data_source_comp: dict[(str, str):[dict[(str, str):list]]] = linkage_problem_io.read_linkage_problems(
    RECORD_LINKAGE_TASKS_PATH, deduplication=False)
linkage_problems = [(k[0], k[1], lp) for k, lp in data_source_comp.items()]
files = [t[0]+"_"+t[1] for t in linkage_problems]
print("number of lp problems {}".format(len(files)))
unsupervised_gold_links = set()
if 'dexter' in linkage_tasks_dir:
    entities, _, _ = reader.read_data(file_name)
    gold_clusters = reader.generate_gold_clusters(entities)
    gold_links = metrics.generate_links(gold_clusters)
    data_sources_dict, data_sources_headers = reader.transform_to_data_sources(entities)
elif 'wdc_computer' in linkage_tasks_dir:
    train_tp_links, train_tn_links, test_tp_links, test_tn_links = wdc_reader.read_wdc_links(train_pairs,
                                                                                             test_pairs)
    gold_links = set()
    gold_links.update(train_tp_links)
    gold_links.update(test_tp_links)
elif 'wdc_almser' in linkage_tasks_dir or 'music_almser' in linkage_tasks_dir:
    gold_links = set()
    train_tp_links, train_tn_links, test_tp_links, test_tn_links, unsup_train_tp_links, unsup_train_tn_links = (
        almser_linkage_reader.read_wdc_links(train_pairs, test_pairs))
    gold_links.update(train_tp_links)
    gold_links.update(test_tp_links)
    unsupervised_gold_links.update(unsup_train_tp_links)
if 'dexter' in linkage_tasks_dir:
    solved_problems, integrated_sources, unsolved_problems = split_linkage_problem_tasks(linkage_problems,
                                                                                     split_ratio=0.5, is_shuffle=True)
elif 'wdc_almser' in linkage_tasks_dir or 'music_almser' in linkage_tasks_dir:
    solved_problems = []
    unsolved_problems = []
    integrated_sources = set()
    tps_check = 0
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
    print("number of tps in lps {}".format(tps_check))

from baseline.almser import almser_wrapper



pairs_fv_train = almser_wrapper.transform_linkage_problems_to_df(solved_problems, fv_splitter, gold_links, unsupervised_gold_links)
pairs_fv_test = almser_wrapper.transform_linkage_problems_to_df(unsolved_problems, fv_splitter, gold_links, unsupervised_gold_links)

from ALMSER_EXP import *
from ALMSER_log import *

almser_path = os.path.join(MAIN_PATH, output_path)

if (query_strategy == 'almser_group'):
    try:
        rltd = pd.read_csv(almser_path + "/heatmap.csv", index_col=0)
        # rltd = pd.read_csv(almser_path+"/task_relatedness.csv", index_col=0)
    except:
        print("ALMSERgroup query strategy needs a relatedness/ heatmap .csv file. Please check.")
else:
    rltd = None

all_nodes_test_match = set(pairs_fv_test[pairs_fv_test.label == 1]['source'].values)
all_nodes_test_match.update(set(pairs_fv_test[pairs_fv_test.label == 1]['target'].values))

all_nodes_train_match = set(pairs_fv_train[pairs_fv_train.label == 1]['source'].tolist())
all_nodes_train_match.update(set(pairs_fv_train[pairs_fv_train.label == 1]['target'].tolist()))

# print("Intersection:", all_nodes_train_match.intersection(all_nodes_test_match))


unique_source_pairs = files
results_concat = pd.DataFrame()
results_all = pd.DataFrame()

for run in range(runs):
    print("RUN %i" % run)
    almser_exp = ALMSER_EXP(pairs_fv_train, pairs_fv_test, unique_source_pairs, max_queries, 'rf',
                            query_strategy, fv_splitter, rltd, bootstrap=True, details=False, batch_size=ACTIVE_LEARNING_ITERATION_BUDGET)

    almser_exp.run_AL(True)

    results_concat = pd.concat((results_concat, (almser_exp.results[
        ['P_model', 'R_model', 'F1_model_micro', 'F1_model_macro', 'F1_model_macro_corrected', 'F1_model_micro_boot', 'tps_boost_graph',
         'fps_boost_graph', 'fns_boost_graph','F1_model_micro_boost_graph',
         'F1_model_macro_boost_graph','F1_model_macro_boost_graph_corrected', 'run_time']])))


results_concat_by_row_index = results_concat.groupby(results_concat.index)
results_concat_mean = results_concat_by_row_index.mean(numeric_only=False)
results_concat_std = results_concat_by_row_index.apply(np.std)
results_concat_sum = results_concat_by_row_index.sum(numeric_only=False)
results_all['P'] = results_concat_mean['P_model']
results_all['P_std'] = results_concat_std['P_model']
results_all['R'] = results_concat_mean['R_model']
results_all['R_std'] = results_concat_std['R_model']
results_all['F1_micro'] = results_concat_mean['F1_model_micro']
results_all['F1_micro_std'] = results_concat_std['F1_model_micro']
results_all['F1_macro'] = results_concat_mean['F1_model_macro']
results_all['F1_macro_std'] = results_concat_std['F1_model_macro']
results_all['F1_macro_corrected'] = results_concat_mean['F1_model_macro_corrected']
results_all['F1_macro_corrected_std'] = results_concat_std['F1_model_macro_corrected']
results_all['F1_micro_boot'] = results_concat_mean['F1_model_micro_boot']
results_all['F1_micro_boot_std'] = results_concat_std['F1_model_micro_boot']
results_all['P_model_micro_boost_graph_corrected'] = results_concat_sum['tps_boost_graph']/(results_concat_sum['fps_boost_graph']+results_concat_sum['tps_boost_graph'])
results_all['R_model_micro_boost_graph_corrected'] = results_concat_sum['tps_boost_graph']/(results_concat_sum['fns_boost_graph']+results_concat_sum['tps_boost_graph'])
results_all['F1_model_micro_boost_graph_corrected'] = 2*results_all['P_model_micro_boost_graph_corrected']*results_all['R_model_micro_boost_graph_corrected']/(results_all['P_model_micro_boost_graph_corrected']+results_all['R_model_micro_boost_graph_corrected'])
results_all['F1_model_micro_boost_graph'] = results_concat_mean['F1_model_micro_boost_graph']
results_all['F1_model_micro_boost_graph_std'] = results_concat_std['F1_model_micro_boost_graph']
results_all['F1_model_macro_boost_graph'] = results_concat_mean['F1_model_macro_boost_graph']
results_all['F1_model_macro_boost_graph_std'] = results_concat_std['F1_model_macro_boost_graph']
results_all['F1_model_macro_boost_graph_corrected'] = results_concat_mean['F1_model_macro_boost_graph_corrected']
results_all['F1_model_macro_boost_graph_corrected_std'] = results_concat_std['F1_model_macro_boost_graph_corrected']
results_all['run_time'] = results_concat_mean['run_time']
#write results
from datetime import datetime

now = datetime.now()
timestamp= now.strftime("%d_%m_%H_%M")
filename = "%i_runs_%i_iter_%s_%s" %(runs,max_queries,query_strategy,timestamp)

#log files
almser_exp.results.to_csv(os.path.join(output_path, filename+"_ALL.csv"), index=False)
almser_exp.labeled_set.to_csv(os.path.join(output_path,filename+"_LABELED_SET_INFO.csv"), index=False)
almser_exp.informants_eval.to_csv(os.path.join(output_path,filename+"_INFORMANTS_EVAL.csv"), index=False)
almser_exp.log.log_info.to_csv(os.path.join(output_path,filename+"_LOG_INFO.csv"), index=False)

#actual results
results_all.to_csv(os.path.join(output_path, filename+".csv"), index=False)

file_name_overall = "almser_res"
almser_res = results_all[['F1_model_micro_boost_graph_std', 'F1_model_micro_boost_graph', 'run_time']]
almser_res = almser_res.assign(budget=max_queries)
almser_res = almser_res.assign(batch=ACTIVE_LEARNING_ITERATION_BUDGET)
almser_res.to_csv(os.path.join(output_path, file_name_overall+".csv"), index=False, mode='a')