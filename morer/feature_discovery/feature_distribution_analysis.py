import numpy as np

from morer.feature_discovery.selection.feature_selector import FeatureSelector

STD = 'std'
SKEW = 'skewness'


def get_additional_features(candidate_features: np.ndarray, criterion: str, selection: FeatureSelector, **kwargs):
    if criterion == STD:
        importance_values = get_additional_features_std(candidate_features)
    elif criterion == SKEW:
        importance_values = get_additional_features_skewness(candidate_features)
    return selection.select_features(importance_values)


def get_additional_features_skewness(candidate_features: np.ndarray):
    mean = np.mean(candidate_features, axis=0)
    median = np.median(candidate_features, axis=0)
    diff_stat = mean - median
    return diff_stat


def get_additional_features_std(candidate_features: np.ndarray):
    std = np.std(candidate_features, axis=0)
    return std
