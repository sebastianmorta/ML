import pandas as pd
import numpy as np

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class RandomSubspaceEnsemble(BaseEnsemble, ClassifierMixin):

    def __init__(self, base_estimator=None, n_estimators=10, n_subspace_features=5, hard_voting=True,
                 random_state=None):

        self.base_estimator = base_estimator

        self.n_estimators = n_estimators

        self.n_subspace_features = n_subspace_features

        self.hard_voting = hard_voting

        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        self.n_features = X.shape[1]



        self.subspaces = np.random.randint(0, self.n_features, (self.n_estimators, self.n_subspace_features))


        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.ensemble_.append(clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))

        return self

    def predict(self, X):

        check_is_fitted(self, "classes_")

        X = check_array(X)

        if X.shape[1] != self.n_features:
            raise ValueError("number of features does not match")

        if self.hard_voting:

            pred_ = []

            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X[:, self.subspaces[i]]))

            pred_ = np.array(pred_)

            prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)

            return self.classes_[prediction]

        else:

            esm = self.ensemble_support_matrix(X)

            average_support = np.mean(esm, axis=0)

            prediction = np.argmax(average_support, axis=1)

            return self.classes_[prediction]

    def ensemble_support_matrix(self, X):

        probas_ = []
        for i, member_clf in enumerate(self.ensemble_):
            probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
        return np.array(probas_)
