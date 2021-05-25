import pandas as pd
import numpy as np
import sklearn

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class RandomSubspaceEnsemble(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, n_subspace_features=5, hard_voting=True,
                 random_state=None):
        # Klasyfikator bazowy
        self.base_estimator = base_estimator
        # Liczba klasyfikatorow
        self.n_estimators = n_estimators
        # Liczba cech w jednej podprzestrzeni
        self.n_subspace_features = n_subspace_features
        # Tryb podejmowania decyzji
        self.hard_voting = hard_voting
        # Ustawianie ziarna losowosci
        self.random_state = random_state
        np.random.seed(self.random_state)


class BaggingEnsemble(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, hard_voting=True):
        print('h')

