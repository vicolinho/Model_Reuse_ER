from abc import ABC

import numpy as np


class FeatureSelector(ABC):

    def __init__(self):
        pass

    def select_features(self, importance_features: np.ndarray):
        return [i for i in range(importance_features.shape[0])]

