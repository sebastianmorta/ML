import numpy as np
from matplotlib import pyplot
from scipy.stats import rankdata, ranksums, ttest_rel
from tabulate import tabulate
from Experiment import Ensemble

en = Ensemble()


def printResultBy(axis_dim, res):
    result = []
    d = "res["
    for i in range(axis_dim):
        d += ":, "
    d += "idx]"
    # print(d)
    # arr = np.array()
    for idx in range(res.shape[axis_dim]):
        # print(idx)
        arr = eval(d)
        result.append(arr)
        # print(arr, arr.shape)
    return result


# a=np.arange(48).reshape(2,2,3,4)
# scores = printResultBy(4, scr)
# print(scr.shape)


#
# for i in range(4):
def calculateStatistics():
    scr = np.load('results.npy')
    # clf_base, method, data, fold, est_qty
    base_clf = ['GNB', 'CART']
    estim_qty = 5
    base_clf_idx = 0
    by_estimators_amount = printResultBy(4, scr)
    for estim in by_estimators_amount:

        # clf_base, method, data, fold

        print("Estimators quantity: ", estim_qty)
        estim_qty += 5
        clfs = printResultBy(0, estim)
        for clf in clfs:
            print("\n\nBase clf: ", base_clf[base_clf_idx])
            base_clf_idx += 1
            if base_clf_idx == 2:
                base_clf_idx = 0
            # method, data, fold,
            # print("by folds")
            # print("\nScores:\n", scores.shape)
            Wilcoxon(clf)
            T_student(clf, clfs)


# calculateStatistics()

def Wilcoxon(clf):
    mean_scores = np.mean(clf, axis=2).T
    # print("\nMean scores:\n", mean_scores)

    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    # print("\nRanks:\n", ranks)

    mean_ranks = np.mean(ranks, axis=0)
    # mean_ranks =np.mean(mean_ranks_tmp,axis=0)

    print("\nMean ranks:\n", mean_ranks)
    print("Random Subspace Ensemble, AdaBoost, Bagging")

    alfa = .05
    w_statistic = np.zeros((len(en.methods), len(en.methods)))
    p_value = np.zeros((len(en.methods), len(en.methods)))

    for i in range(len(en.methods)):
        for j in range(len(en.methods)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    headers = list(en.methods.keys())
    names_column = np.expand_dims(np.array(list(en.methods.keys())), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(en.methods), len(en.methods)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(en.methods), len(en.methods)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)
    print("----------------------------------------------------------------------\n\n")


def T_student(clf, scores):
    alfa = .05
    t_statistic = np.zeros((len(en.methods), len(en.methods)))
    p_value = np.zeros((len(en.methods), len(en.methods)))
    for i in range(len(en.methods)):
        for j in range(len(en.methods)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
    headers = list(en.methods.keys())
    names_column = np.expand_dims(np.array(list(en.methods.keys())), axis=1)
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
    advantage = np.zeros((len(en.methods), len(en.methods)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)
    significance = np.zeros((len(en.methods), len(en.methods)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)
    print("----------------------------------------------------------------------\n\n")






# converting string data to to float. Takes dataset in the form of dataframe as argument.
    def string_to_number(self, dataset):
        le = preprocessing.LabelEncoder()
        datatypes = dataset.dtypes
        for i in range(len(datatypes)):
            if datatypes[i] == 'object':
                dataset[i] = le.fit_transform(dataset[i])
        return dataset























































































