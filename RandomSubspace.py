import pandas as pd
import numpy as np

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

    def fit(self, X, y):
        X, y = check_X_y(X, y)  # Sprawdzenie czy X i y maja wlasciwy ksztalt

        self.classes_ = np.unique(y)  # Przehowywanie nazw klas

        self.n_features = X.shape[1]  # Zapis liczby atrybutow

        if self.n_subspace_features > self.n_features:  # Czy liczba cech w podprzestrzeni jest mniejsza od calkowitej liczby cech
            raise ValueError("Number of features in subspace higher than number of features.")

        self.subspaces = np.random.randint(0, self.n_features, (
        self.n_estimators, self.n_subspace_features))  # Wylosowanie podprzestrzeni cech

        self.ensemble_ = []  # Wyuczenie nowych modeli i stworzenie zespolu
        for i in range(self.n_estimators):
            self.ensemble_.append(clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))

        return self
