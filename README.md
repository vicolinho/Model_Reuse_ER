# Model Reuse via Feature Distribution Analysis for Incremental Entity Resolution

Entity resolution (ER) is a fundamental task in data integration that enables insights from heterogeneous data sources. 
The primary challenge of ER lies in classifying record pairs as matches or non-matches, 
which in multi-source ER (MS-ER) scenarios can become complicated due to data source heterogeneity and
scalability issues. Existing methods for MS-ER generally require labeled
record pairs, and such methods fail to effectively reuse models across multiple ER tasks. 

We propose _MoRER_ (Model Repositories for Entity Resolution), a novel method for building a model repository
consisting of classification models that solve ER problems. By leveraging feature distribution analysis, 
_MoRER_ clusters similar ER tasks thereby enabling the effective initialization of a model repository
with a moderate labeling effort. 

Experimental results on three multi-source datasets demonstrate that 
_MoRER_ achieves comparable or better results to methods which have label-limited budges, such
as active learning and transfer learning approaches, while outperforming unsupervised and 
self-supervised approaches that utilize large pre-trained language models. When compared to supervised
transformer-based methods, MoRER achieves comparable or better results depending on the training data size. 
Importantly, _MoRER_ is the first method for building a model repository for ER problems, 
facilitating the continuous integration of new data sources by reducing the need for generating new training data.


## Workflow

![](workflow.png)


## Datasets
| Name         | Source                                                                                                  | Description                                                                                                                                                                                                               |
|--------------|---------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dexter       | <a href='https://cloud.scadsai.uni-leipzig.de/index.php/s/RkoSzpdwkyYc87s'> Link </a>                   | camera data set comprises similarity graphs for 21023 camera product specifications from 23 data sources. The original data set was used in the Sigmod Programming Contest. We use only the records with their properties |
| wdc_almser   | <a href='http://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/almser_gen_data/'> Link </a> | provided data sets consisting of pairwise features from the Almser publication [1]                                                                                                                                        |
| music_almser | <a href='http://data.dws.informatik.uni-mannheim.de/benchmarkmatchingtasks/almser_gen_data/'> Link </a>                                                                      | provided data sets consisting of pairwise features from the Almser publication [1]                                                                                                                                        |

## Linkage Problem Generation

### Dexter
The generation is tailored for the dexter data set which is saved in the Gradoop csv format.
The following call generate similarity vectors for each data source pair and save the result as a pickle file consisting of 
a dictionary where the key is the data source pair and the value a dictionary of record pairs as key 
and feature vectors as value. 

`python morer/data_set_generation/linkage_generation_dexter.py -d datasets/(dataset_dir) -o data/linage_problems/(lp_path)`

### Almser datasets
To use the raw feature files provided from the Almser publication, we transform them to our format where each data source pair is saved in a dictionary 
with the similarity feature vector dictionary as value.

`python record_linkage/almser_linkage_reader.py -ff data/linkage_problems/music_almser/source_pairs
    -tp data/linkage_problems/selected_data_set/train_pairs_fv.csv 
    -gs data/linkage_problems/selected_data_set/test_pairs_fv.csv 
    -lo data/linkage_problems/selected_data_set`


## Run MoRER

run MoRER on the dexter data set with bootstrap active learning using a budget of 1000.

`python ./morer/reuse/main.py -d ./datasets/dexter/DS-C0/SW_0.3 -l ./data/linkage_problems/dexter/ -mg bootstrap -tb 1000`

run MoRER on the dexter data set with bootstrap active learning using a budget of 1000 and retraining.

`python ./morer/reuse/main.py -d ./datasets/dexter/DS-C0/SW_0.3 -l ./data/linkage_problems/dexter/ -mg bootstrap -tb 1000 -rec True -uns_ratio 0.25`

### Parameters

| Name                               | Description                                                                                                  | Options                                                                       |
|------------------------------------|--------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `--train_pairs`<br>`-tp`           | Train pairs (input file containing training data).                                                           | -                                                                             |
| `--test_pairs`<br>`-gs`            | Test pairs (input file containing test data).                                                                | -                                                                             |
| `--linkage_tasks_dir`<br>`-l`      | Directory for linkage problems.                                                                              | -                                                                             |
| `--statistical_test`<br>`-s`       | Statistical test for comparing linkage problems.                                                             | default: `ks_test`<br> `wasserstein_distance`, `calculate_psi`                |
| `--comm_detect`<br>`-cd`           | Community detection algorithm to use.                                                                        | default:`leiden`<br>`louvain`, `girvan_newman`,<br>`label_propagation_clustering` |
| `--model_generation`<br>`-mg`      | Model generation algorithm to use.                                                                           | default: `bootstrap`<br> `almser`,`supervised`                          |
| `--min_budget`<br>`-mb`            | Minimum budget for each cluster.                                                                             | default: `50`                                                                 |
| `--total_budget`<br>`-tb`          | Total budget for the entire process.                                                                         | default: `1000`                                                               |
| `--batch_size`<br> `-b`            | Batch size for active learning.                                                                              | default: `5`                                                                  |
| `--is_recluster`<br> `-rec`        | is recluster                                                                                                 | default: `False`                                                              |
| `--unsolved_ratio`<br>`-uns_ratio` | ratio threshold t_cov of the number of problems not being used for training and all problems in a cluster.   | default: `0.25`                                                               |
| `--budget_retrain`<br> `-b_rt`     | label budget for retraining step if only new problems are in the cluster                                     | default: `250`                                                                |

## References

[1] Anna Primpeli, Christian Bizer:
Graph-Boosted Active Learning for Multi-source Entity Resolution. ISWC 2021: 182-199
