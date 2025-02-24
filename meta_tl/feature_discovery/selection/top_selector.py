import numpy as np

from meta_tl.feature_discovery.selection.feature_selector import FeatureSelector


class TopSelector(FeatureSelector):

    def __init__(self, top_feature):
        super().__init__()
        self.top_feature = top_feature

    def select_features(self, importance_features: np.ndarray):
        return importance_features.argsort()[-self.top_feature:][::-1]
