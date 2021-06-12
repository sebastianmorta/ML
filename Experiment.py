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
    def __init__(self, datasets=os.listdir('datasets2')):
        self.n_datasets = 20
        # self.n_datasets = len(datasets)
        self.n_splits = 5
        self.n_repeats = 2
        self.rskf = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=1234)
        self.datasets = datasets
        self.clfs = {
            'GNB': GaussianNB(),
            'CART': DecisionTreeClassifier(random_state=1234),
        }
        self.methods = {
            "AdaBoost": AdaBoostClassifier,
            "Bagging": BaggingClassifier,
            "RndSpcEnmbl": RandomSubspaceEnsemble,

        }
        self.n_estimators = [5, 10, 15]
        self.scores = np.zeros((len(self.clfs), len(self.methods), self.n_datasets, self.n_splits * self.n_repeats,
                                len(self.n_estimators)))

    def skip_lines(self, path: 'str') -> int:
        num = 0
        if path.endswith('.dat'):
            with open(path) as f:
                for line in f.readlines():
                    num += 1
                    if '@data' in line:
                        return num
        else:
            return 1

    def drop_rows_with_missing_values(self, dataset) -> pd.DataFrame:
        dataset.dropna(axis=0, how='any')
        return dataset

    def drop_insufficient_rows(self, dataset, class_list) -> pd.DataFrame:
        minimum_amount_of_fata = len(dataset.index) * 0.001
        if minimum_amount_of_fata < 10:
            minimum_amount_of_fata = 10
        for c in class_list:
            if dataset.iloc[:, -1:].value_counts()[c] < minimum_amount_of_fata:
                dataset = dataset[dataset.iloc[:, -1] != c]
        return dataset

    def prepare_data(self, dataset_dir_path, dataset_name) -> pd.DataFrame:
        dataset_path = os.path.join(dataset_dir_path, dataset_name)
        dataset = pd.read_csv(dataset_path, header=None)
        classes_list = dataset.iloc[:, -1].unique()
        dataset = self.drop_rows_with_missing_values(dataset)
        dataset = self.drop_insufficient_rows(dataset, classes_list)
        dataset = self.string_to_number(dataset)
        return dataset

    def string_to_number(self, dataset):
        le = preprocessing.LabelEncoder()
        datatypes = dataset.dtypes
        for i in range(len(datatypes)):
            if datatypes[i] == 'object':
                dataset[i] = le.fit_transform(dataset[i])
        return dataset

    def changeLabels(self, labels, low):
        for i in range(len(labels)):
            labels[i] = labels[i] + low
        return labels

    def changeLabels2(self, labels, high):
        for i in range(len(labels)):
            labels[i] = labels[i] - high
        return labels

    def makeResult(self):
        for data_id, dataset in enumerate(self.datasets):

            # dataset = np.genfromtxt("datasets/%s" % (dataset), delimiter=",")
            dataset_path = "datasets2"
            print(colored(dataset, 'red'))
            dataset = self.prepare_data(dataset_path, dataset)
            dataset = self.string_to_number(dataset)
            dataset = dataset.to_numpy()

            X = dataset[:, :-1]
            y = dataset[:, -1].astype(int)
            for method_id, method_name in enumerate(self.methods):
                for fold_id, (train, test) in enumerate(self.rskf.split(X, y)):
                    for clf_id, clf_name in enumerate(self.clfs):
                        # clf_base = clone(self.clfs[clf_name])
                        for est_id, est_amount in enumerate(self.n_estimators):
                            method = self.methods[method_name](base_estimator=clone(self.clfs[clf_name]),
                                                               n_estimators=est_amount,
                                                               random_state=123)

                            method.fit(X[train], y[train])
                            y_pred = method.predict(X[test])
                            # row_ix = where(y == classmethod)
                            # pyplot.scatter(X[row_ix,0], X[row_ix,1])
                            # pyplot.show()
                            self.scores[clf_id, method_id, data_id, fold_id, est_id] = accuracy_score(y[test], y_pred)
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
