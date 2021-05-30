import os

from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from RandomSubspace import RandomSubspaceEnsemble


class Ensemble:
    def __init__(self, datasets=os.listdir('datasets')):
        self.n_datasets = len(datasets)
        self.n_splits = 5
        self.n_repeats = 2
        self.rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=1234)
        self.datasets = datasets
        self.clfs = {
            'GNB': GaussianNB(),
            'CART': DecisionTreeClassifier(random_state=1234),
        }
        self.methods = {
            "RndSpcEnmbl": RandomSubspaceEnsemble,
            "AdaBoost": AdaBoostClassifier,
            "Bagging": BaggingClassifier
        }
        self.scores = np.zeros((len(self.clfs), self.n_datasets, self.n_splits * self.n_repeats, len(self.methods)))
        self.n_estimators = [5, 10, 15]

    def makeResult(self):
        for data_id, dataset in enumerate(self.datasets):
            dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
            for method_id, method_name in enumerate(self.methods):
                X = dataset[:, :-1]
                y = dataset[:, -1].astype(int)
                for fold_id, (train, test) in enumerate(self.rskf.split(X, y)):
                    for clf_id, clf_name in enumerate(self.clfs):
                        clf_base = clone(self.clfs[clf_name])
                        for est in self.n_estimators:
                            method = self.methods[method_name](base_estimator=self.clfs[clf_name], n_estimators=est,
                                                               random_state=123)

                            method.fit(X[train], y[train])
                            y_pred = method.predict(X[test])
                            self.scores[clf_id, data_id, fold_id, method_id] = accuracy_score(y[test], y_pred)
                        # print(accuracy_score(y[test], y_pred))
        np.save('results', self.scores)
        # print(self.scores)


a = Ensemble()
a.makeResult()
# datasets = os.listdir('datasets')
# clfs = {
#     'GNB': GaussianNB(),
#     'CART': DecisionTreeClassifier(random_state=1234),
# }
#
# methods = {
#     "clf1": RandomSubspaceEnsemble,
#     "clf2": AdaBoostClassifier,
#     "clf3": BaggingClassifier
# }
#
# # X, y = load_iris(return_X_y=True)
# n_datasets = len(datasets)
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
#
# scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats, len(methods)))
# for data_id, dataset in enumerate(datasets):
#     dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
#     for method_id, method_name in enumerate(methods):
#         X = dataset[:, :-1]
#         y = dataset[:, -1].astype(int)
#         for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#             for clf_id, clf_name in enumerate(clfs):
#                 clf_base = clone(clfs[clf_name])
#                 method = methods[method_name](base_estimator=clfs[clf_name], n_estimators=10, random_state=123)
#
#                 clf_base.fit(X[train], y[train])
#                 y_pred = clf_base.predict(X[test])
#                 scores[clf_id, data_id, fold_id, method_id] = accuracy_score(y[test], y_pred)
#                 print(accuracy_score(y[test], y_pred))
# np.save('results', scores)
# print(scores)
