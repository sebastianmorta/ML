import numpy as np
from scipy.stats import rankdata, ranksums
from tabulate import tabulate
from tmp import Ensemble

en = Ensemble()


class Axis:
    def __init__(self, axis_name, axis_dim):
        self.axis_name = axis_name
        self.axis_dim = axis_dim


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


scr = np.load('results.npy')
# a=np.arange(48).reshape(2,2,3,4)
scores = printResultBy(4, scr)
print(scr.shape)


#
# for i in range(4):
def calculateStatistics():
    # clf_base, method, data, fold, est_qty
    by_estimators_amount = printResultBy(4, scr)
    for estim in by_estimators_amount:
        # clf_base, method, data, fold

        print("new estimator")
        score = printResultBy(0, estim)
        for scores in score:
            print("new clf")
            # method, data, fold,
            # print("by folds")
            # print("\nScores:\n", scores.shape)

            mean_scores = np.mean(scores, axis=1).T
            # print("\nMean scores:\n", mean_scores)

            ranks = []
            for ms in mean_scores:
                ranks.append(rankdata(ms).tolist())
            ranks = np.array(ranks)
            # print("\nRanks:\n", ranks)

            mean_ranks = np.mean(ranks, axis=0)
            # print("\nMean ranks:\n", mean_ranks)

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
            print("\nAdvantage:\n", advantage_table)

            significance = np.zeros((len(en.methods), len(en.methods)))
            significance[p_value <= alfa] = 1
            significance_table = tabulate(np.concatenate(
                (names_column, significance), axis=1), headers)
            print("\nStatistical significance (alpha = 0.05):\n", significance_table)
            print("----------------------------------------------------------------------")


calculateStatistics()
