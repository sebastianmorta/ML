import numpy as np
import pandas as pd
from sklearn import datasets

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from RandomSubspace import RandomSubspaceEnsemble
from scipy.io.arff import loadarff

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

number_of_estimators = 50
# estimator = MultinomialNB()
clfs = {
    'GNB': GaussianNB(),
    'BNB': BernoulliNB(),
    'CART': DecisionTreeClassifier(random_state=42),
    'MNB': MultinomialNB()
}
estimator = GaussianNB()


# dataset = datasets.load_wine()


def dataSetReader(file):
    dataset=file
    dataset = np.genfromtxt("%s.csv" % (file), delimiter=",")
    X = dataset[:, :-1]
    Y = dataset[:, -1].astype(int)
    print(dataset)
    return X, Y


# dataSetReader()
# X = dataset.data
# y = dataset.target
# print(X)
# print('-----')
# print(y)
# print('=======')
ada_scores = []
bagging_scores = []
rnd_subspace_scores = []
list_of_datasets = os.listdir('datasets')
# list_of_datasets = ["iris", "balcnce", "contraceptive", "thyroid", "splice", "tae", "car", "lymphography", "vehicle", "automobile", "glass", "census", "newthyroid", "wine", "cleveland", "flare", "satimage", "yeast", "", "", "", ]







# X, y = dataSetReader('datasets/australian')
for i in range(len(list_of_datasets)):
    for clf_id, clf_name in enumerate(clfs):
        for k in range(len(estimators)):
            X, y = dataSetReader(list_of_datasets[i])
            # print(X)
            # print('-----')
            # print(y)
            print("Current dataset ", str(list_of_datasets[i]))
            print("Total number of features is", X.shape[1])
            print("\n")
            n_splits = 5
            n_repeats = 10
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=123)

            clf = RandomSubspaceEnsemble(base_estimator=clfs[clf_name], n_estimators=estimators[k], n_subspace_features=5,
                                         hard_voting=True, random_state=123)
            clf2 = AdaBoostClassifier(base_estimator=clfs[clf_name], n_estimators=estimators[k], random_state=123)

            clf3 = BaggingClassifier(base_estimator=clfs[clf_name], n_estimators=estimators[k], random_state=123)
            scores = []
            for train, test in rskf.split(X, y):
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores.append(accuracy_score(y[test], y_pred))
            # print("Hard voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
            rnd_subspace_scores.append(np.mean(scores))

            scores = []
            for train, test in rskf.split(X, y):
                clf.fit(X[train], y[train])
                y_pred = clf.predict(X[test])
                scores.append(accuracy_score(y[test], y_pred))
            # print("Soft voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
            rnd_subspace_scores.append(np.mean(scores))

            scores = []
            for train, test in rskf.split(X, y):
                clf2.fit(X[train], y[train])
                y_pred = clf2.predict(X[test])
                scores.append(accuracy_score(y[test], y_pred))
            # print("Ada - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
            ada_scores.append(np.mean(scores))

            scores = []
            for train, test in rskf.split(X, y):
                clf3.fit(X[train], y[train])

                y_pred = clf3.predict(X[test])
                scores.append(accuracy_score(y[test], y_pred))
            # print("Bagging - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
            bagging_scores.append(np.mean(scores))

            print('Bagging scores: ', bagging_scores)
            print('Ada scores: ', ada_scores)
            print('RND subspace scores: ', rnd_subspace_scores)

            print('')
            print('klasyfikator bazowy', clfs[clf_name])
            print('liczba estymatorów', str(estimators[k]))
            print('Średnia bag:', np.mean(bagging_scores))
            print('Średnia ada:', np.mean(ada_scores))
            print('Średnia rnd:', np.mean(rnd_subspace_scores))
            print('\n\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
