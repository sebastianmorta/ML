import numpy as np
import pandas as pd
from sklearn import datasets

from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from RandomSubspace import RandomSubspaceEnsemble
from scipy.io.arff import loadarff

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

number_of_estimators = 50
# estimator = MultinomialNB()
estimator = GaussianNB()
dataset = datasets.load_wine()


# class RandomSubspaceEnsemble(BaseEnsemble, ClassifierMixin):
#
#     def __init__(self, base_estimator=None, n_estimators=10, n_subspace_features=5, hard_voting=True,
#                  random_state=None):
#         # Klasyfikator bazowy
#         self.base_estimator = base_estimator
#         # Liczba klasyfikatorow
#         self.n_estimators = n_estimators
#         # Liczba cech w jednej podprzestrzeni
#         self.n_subspace_features = n_subspace_features
#         # Tryb podejmowania decyzji
#         self.hard_voting = hard_voting
#         # Ustawianie ziarna losowosci
#         self.random_state = random_state
#         np.random.seed(self.random_state)
#
#     def fit(self, X, y):
#         X, y = check_X_y(X, y)
#         self.classes_ = np.unique(y)
#         self.n_features = np.unique(y)
#         self.n_features = X.shape[1]
#
#         if self.n_subspace_features > self.n_features:
#             raise ValueError("Number")
#
#         self.subspaces = np.random.randint(0, self.n_features, (self.n_estimators, self.n_subspace_features))
#
#         # Wyuczenie nowych modeli i stworzenie zespolu
#         self.ensemble_ = []
#         for i in range(self.n_estimators):
#             self.ensemble_.append(clone(self.base_estimator).fit(X[:, self.subspaces[i]], y))
#
#         return self
#
#     def predict(self, X):
#         # Sprawdzenie czy modele sa wyuczone
#         check_is_fitted(self, "classes_")
#         # Sprawdzenie poprawnosci danych
#         X = check_array(X)
#         # Sprawdzenie czy liczba cech się zgadza
#         if X.shape[1] != self.n_features:
#             raise ValueError("number of features does not match")
#
#         if self.hard_voting:
#             # Podejmowanie decyzji na podstawie twardego glosowania
#             pred_ = []
#             # Modele w zespole dokonuja predykcji
#             for i, member_clf in enumerate(self.ensemble_):
#                 pred_.append(member_clf.predict(X[:, self.subspaces[i]]))
#             # Zamiana na miacierz numpy (ndarray)
#             pred_ = np.array(pred_)
#             # Liczenie glosow
#             prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
#             # Zwrocenie predykcji calego zespolu
#             return self.classes_[prediction]
#
#         else:
#             # Podejmowanie decyzji na podstawie wektorow wsparcia
#             esm = self.ensemble_support_matrix(X)
#             # Wyliczenie sredniej wartosci wsparcia
#             average_support = np.mean(esm, axis=0)
#             # Wskazanie etykiet
#             prediction = np.argmax(average_support, axis=1)
#             # Zwrocenie predykcji calego zespolu
#             return self.classes_[prediction]
#
#     def ensemble_support_matrix(self, X):
#         # Wyliczenie macierzy wsparcia
#         probas_ = []
#         for i, member_clf in enumerate(self.ensemble_):
#             probas_.append(member_clf.predict_proba(X[:, self.subspaces[i]]))
#         return np.array(probas_)


# dat_file = r"D:\Python\UM\iris1.dat"
# text = ''
# with open(dat_file, 'r') as file:
#     text = file.read()
#     print(text)
def dataSetReader():
    data = []
    with open('datasets\\iris.dat') as file:
        n, m = file.readline().split()
        for i in file:
            if not i.isspace():
                data.append([int(x) for x in i.split()])
    return data, int(n), int(m)







dataSetReader()
X = dataset.data
y = dataset.target
print("Total number of features is", X.shape[1])

n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=123)

clf = RandomSubspaceEnsemble(base_estimator=estimator, n_estimators=number_of_estimators, n_subspace_features=5,
                             hard_voting=True, random_state=123)
clf2 = AdaBoostClassifier(base_estimator=estimator, n_estimators=number_of_estimators, random_state=123)

clf3 = BaggingClassifier(base_estimator=estimator, n_estimators=number_of_estimators, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Hard voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Soft voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

scores = []
for train, test in rskf.split(X, y):
    clf2.fit(X[train], y[train])
    y_pred = clf2.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Ada - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

scores = []
for train, test in rskf.split(X, y):
    clf3.fit(X[train], y[train])
    y_pred = clf3.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Bagging - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
