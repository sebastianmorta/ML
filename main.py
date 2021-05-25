import pandas as pd
import numpy as np
import sklearn

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class RandomSubspaceEnsemble(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, n_subspace_features=5, hard_voting=True,
                 random_state=None):
        self.base_estimator = base_estimator  # Klasyfikator bazowy

        self.n_estimators = n_estimators  # Liczba klasyfikatorow

        self.n_subspace_features = n_subspace_features  # Liczba cech w jednej podprzestrzeni

        self.hard_voting = hard_voting  # Tryb podejmowania decyzji

        self.random_state = random_state  # Ustawianie ziarna losowosci
        np.random.seed(self.random_state)


class BaggingEnsemble(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, hard_voting=True):
        print('h')
