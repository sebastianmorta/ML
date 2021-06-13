import csv

import numpy as np
from matplotlib import pyplot
from scipy.stats import rankdata, ranksums, ttest_rel
from tabulate import tabulate
from Experiment import Ensemble
from pipenv.vendor.vistir.termcolors import colored

en = Ensemble()

def wilcoxon2(estim):
    # clf_base, method, data, fold
    print(colored("WILKOXON2", 'magenta'))
    # print(clf)
    mean_scores = np.mean(estim, axis=3).T
    # clf_base, method, data
    # print("\nMean scores:\n", mean_scores)
    # print(mean_scores)
    ranks = []
    # for mss in mean_scores:
    #     for ms in mss:
    #         ranks.append(rankdata(ms).tolist())
    # ranks = np.array(ranks)
    # print("\nRanks:\n", ranks)


    #mean_scores = np.mean(mean_scores, axis=0)
    for mr in mean_scores:
        ranks.append(rankdata(mr).tolist())
    ranks = np.array(ranks)
    # print("\nRanks:\n", ranks)


    # mean_ranks =np.mean(mean_ranks_tmp,axis=0)
    #print(mean_ranks.T)
    mean_ranks = np.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks)
    # print("AdaBoost, Bagging, Random Subspace")

    alfa = .05
    w_statistic = np.zeros((len(en.methods)*len(en.clfs), len(en.methods)*len(en.clfs)))
    p_value = np.zeros((len(en.methods)*len(en.clfs), len(en.methods)*len(en.clfs)))

    for i in range(len(en.methods)*len(en.clfs)):
        for j in range(len(en.methods)*len(en.clfs)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])
    headers=[]
    for m in en.methods.keys():
        for c in en.clfs.keys():
            headers.append(m+c)
    # headers = list(en.methods.keys())*2
    names_column = np.expand_dims(np.array(headers), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(en.methods)*len(en.clfs), len(en.methods)*len(en.clfs)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(en.methods)*len(en.clfs), len(en.methods)*len(en.clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    # print(colored("\nStatistical significance (alpha = 0.05):", 'magenta'))
    # print(significance_table)
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)
    print(len(stat_better_table))

def tStudent2(estim):
    # clf_base, method, data, fold
    print(colored("WILKOXON2", 'magenta'))
    # print(clf)
    mean_scores = np.mean(estim, axis=3).T
    # clf_base, method, data
    # print("\nMean scores:\n", mean_scores)
    # print(mean_scores)
    ranks = []
    # for mss in mean_scores:
    #     for ms in mss:
    #         ranks.append(rankdata(ms).tolist())
    # ranks = np.array(ranks)
    # print("\nRanks:\n", ranks)


    #mean_scores = np.mean(mean_scores, axis=0)
    for mr in mean_scores:
        ranks.append(rankdata(mr).tolist())
    ranks = np.array(ranks)
    # print("\nRanks:\n", ranks)


    # mean_ranks =np.mean(mean_ranks_tmp,axis=0)
    #print(mean_ranks.T)
    mean_ranks = np.mean(ranks, axis=0)
    print("\nMean ranks:\n", mean_ranks)
    # print("AdaBoost, Bagging, Random Subspace")

    alfa = .05
    t_statistic = np.zeros((len(en.methods)*len(en.clfs), len(en.methods)*len(en.clfs)))
    p_value = np.zeros((len(en.methods)*len(en.clfs), len(en.methods)*len(en.clfs)))

    for i in range(len(en.methods)*len(en.clfs)):
        for j in range(len(en.methods)*len(en.clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(ranks.T[i], ranks.T[j])
    headers=[]
    for m in en.methods.keys():
        for c in en.clfs.keys():
            headers.append(m+c)
    # headers = list(en.methods.keys())*2
    names_column = np.expand_dims(np.array(headers), axis=1)
    w_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(en.methods)*len(en.clfs), len(en.methods)*len(en.clfs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)

    significance = np.zeros((len(en.methods)*len(en.clfs), len(en.methods)*len(en.clfs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    # print(colored("\nStatistical significance (alpha = 0.05):", 'magenta'))
    # print(significance_table)
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)
    print(len(stat_better_table))
def wilcoxon(clf):
    print(colored("WILKOXON", 'magenta'))
    # print(clf)
    mean_scores = np.mean(clf, axis=2).T
    # print("\nMean scores:\n", mean_scores)
    # print(mean_scores)
    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    # print("\nRanks:\n", ranks)

    mean_ranks = np.mean(ranks, axis=0)
    # mean_ranks =np.mean(mean_ranks_tmp,axis=0)

    print("\nMean ranks:\n", mean_ranks)
    # print("AdaBoost, Bagging, Random Subspace")

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
    # print(colored("\nStatistical significance (alpha = 0.05):", 'magenta'))

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print(colored("Statistically significantly better:\n",  'magenta'))
    print(stat_better_table,)

def tStudent(clf):
    print(colored("T-STUDENT", 'green'))
    mean_scores = np.mean(clf, axis=2).T
    # print("\nMean scores:\n", mean_scores)
    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    # print("\nRanks:\n", ranks)

    mean_ranks = np.mean(ranks, axis=0)
    # mean_ranks =np.mean(mean_ranks_tmp,axis=0)

    # print("\nMean ranks:\n", mean_ranks)
    # print("Random Subspace Ensemble, AdaBoost, Bagging")
    alfa = .05
    t_statistic = np.zeros((len(en.methods), len(en.methods)))
    p_value = np.zeros((len(en.methods), len(en.methods)))
    for i in range(len(en.methods)):
        for j in range(len(en.methods)):
            t_statistic[i, j], p_value[i, j] = ttest_rel(mean_scores[i], mean_scores[j])
    # print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
    headers = list(en.methods.keys())
    names_column = np.expand_dims(np.array(list(en.methods.keys())), axis=1)
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
    advantage = np.zeros((len(en.methods), len(en.methods)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
    # print("\nAdvantage:\n", advantage_table)
    significance = np.zeros((len(en.methods), len(en.methods)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate((names_column, significance), axis=1), headers)
    # print(colored("\nStatistical significance (alpha = 0.05):", 'green'))
    # print(significance_table)
    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print(colored("Statistically significantly better:\n",  'green'))
    print(stat_better_table,)


def saveToSCV(data, first_row, y_label, name):
    with open(name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_row)
        # writer.writerow([["GNB", "CART"], ["GNB", "CART"], ["GNB", "CART"]])
        # y_label2 = []
        # for i in y_label:
        #     elo = os.path.splitext(i)
        #     y_label2.append(elo[0])
        for i in range(len(data)):
            for j in range(3):
                data[i][j] = round(data[i][j], 3)

        for y, d in zip(y_label, data):

            writer.writerow([y]+d.tolist())


# saveToSCV([1,2,3,4,5,6])


def getTotalMeanScores(scores):
    print(colored('==================MEAN=========', 'red'))
    ranks = []
    print("[AdaBoost, Bagging, Random Subspace]")
    # RANKS ALL
    print('\nMEAN RANKS')

    mean_scores = np.mean(scores, axis=4)
    mean_scores = np.mean(mean_scores, axis=0)
    mean_scores = np.mean(mean_scores, axis=2).T
    for mr in mean_scores:
        ranks.append(rankdata(mr).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    print(mean_ranks)
    # MEAN SCORES ALL
    print('\nMEAN SCORES')

    mean_scores2 = np.mean(scores, axis=4)
    mean_scores2 = np.mean(mean_scores2, axis=0)
    mean_scores2 = np.mean(mean_scores2, axis=2)
    mean_scores2 = np.mean(mean_scores2, axis=1).T
    print(colored(mean_scores2, 'red'))

    ###########DATASETS
    print('\n\nDATASETS')

    mean_scores3 = np.mean(scores, axis=4)
    mean_scores3 = np.mean(mean_scores3, axis=0)
    mean_scores3 = np.mean(mean_scores3, axis=2).T
    print(colored(mean_scores3, 'red'))
    # saveToSCV(mean_scores3, ["Dataset", "Ada Boost", "Bagging", "Random Subspace"], os.listdir('datasets2'))
    ##########NUMBER OF ESTIMATORS
    print('\n\nNUMBER OF ESTIMATORS')
    mean_scores4 = np.mean(scores, axis=3)
    mean_scores4 = np.mean(mean_scores4, axis=0)
    mean_scores4 = np.mean(mean_scores4, axis=1).T
    print(colored(mean_scores4, 'red'))
    saveToSCV(mean_scores, ["Estimators Amount", "Ada Boost", "Bagging", "Random Subspace"], [5,10,15],"byest")

    #######BASE CLFS
    print('\n\nBASE CLFS')
    mean_scores5 = np.mean(scores, axis=2)
    mean_scores5 = np.mean(mean_scores5, axis=2)
    mean_scores5 = np.mean(mean_scores5, axis=2)
    print(colored(mean_scores5, 'red'))
    saveToSCV(mean_scores5, ["Base Clf", "Ada Boost", "Bagging", "Random Subspace"],['GNB', 'CART'],"byclf" )



def plotByClf(scr):
    by_estimators_amount = printResultBy(4, scr)
    order = [6, 3, 4, 4, 5, 3, 6, 7, 3, 26, 4, 3, 7, 3, 3, 11, 3, 4, 3, 2]
    estim_qty = 0
    base_clf_idx = 0
    # colors = ["mo--", "go--", 'bo--', 'ro--', 'yo--', 'co--']
    colors = ["mo", "go", 'bo', 'ro', 'yo', 'co']
    x_label = os.listdir('datasets2')
    tmp = []
    for i in x_label:
        elo = os.path.splitext(i)
        tmp.append(elo[0])
    x_label=tmp
    x_label = [y for _, y in sorted(zip(order, x_label))]
    met = ["AdaBoost-GNB", "Bagging-GNB", "RandomSubspace-GNB", "AdaBoost-CART", "Bagging-CART", "RandomSubspace-CART"]

    plt.rcParams.update({'font.size': 16})
    for estim in by_estimators_amount:
        # clf_base, method, data, fold
        estim_qty += 5
        values = []
        clfs = printResultBy(0, estim)
        for clf in clfs:
            #  method, data, fold
            base_clf_idx += 1
            if base_clf_idx == 2:
                base_clf_idx = 0

            val = np.mean(clf, axis=2).T
            val = printResultBy(1, val)
            values += val
        fig, ax = plt.subplots(figsize=(24, 16))

        for v, color, m in zip(values, colors, met):
            v = [y for _, y in sorted(zip(order, v))]
            # print("values",v)
            ax.plot(x_label, v, color, label=m, markersize=14)
        plt.title(f"Estimators Amount - {estim_qty}", size=20)
        plt.xticks(rotation=50)
        plt.xlabel("Datasets", size=16)
        plt.ylabel("Accuracy", size=16)
        plt.legend(prop={'size': 16})
        plt.grid(1, 'major')
        # print("------------------")
        plt.savefig(f"Estimators Amount - {estim_qty}")
        plt.show()


statistics = {
    "WILKOXON": wilcoxon,
    "T_STUDENT": tStudent
}

statistics2 = {
    "WILKOXON": wilcoxon2,
    "T_STUDENT": tStudent2
}


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


def calculateStatistics():
    scr = np.load('results.npy')
    # getTotalMeanScores(scr)
    # clf_base, method, data, fold, est_qty
    ds = os.listdir('datasets2')
    # plotByClf(scr)
    for stat_id, stat_name in enumerate(statistics):
        base_clf = ['GNB', 'CART']
        estim_qty = 5
        base_clf_idx = 0

        # clfs = printResultBy(0, scr)
        # for clf in clfs:
        #     #  method, data, fold,est_qty
        #     print(colored("========= Base clf: " + str(base_clf[base_clf_idx]), 'yellow'))
        #     base_clf_idx += 1
        #     if base_clf_idx == 2:
        #         base_clf_idx = 0
        #     by_estimators_amount = printResultBy(3, clf)
        #     estim_qty = 5
        #     for estim in by_estimators_amount:
        #         #  method, data, fold
        #         print(colored("\n============== Estimators quantity: " + str(estim_qty) + " ==============", 'blue'))
        #         print()
        #         estim_qty += 5
        #         statistics[stat_name](estim)
        by_estimators_amount = printResultBy(4, scr)
        for estim in by_estimators_amount:
            # clf_base, method, data, fold
            print(colored("\n============== Estimators quantity: " + str(estim_qty) + " ==============", 'blue'))
            print()
            estim_qty += 5
            # wilcoxon2(estim)
            # statistics[stat_name](estim)
            clfs = printResultBy(0, estim)
            for clf in clfs:
                #  method, data, fold
                print(colored("========= Base clf: " + str(base_clf[base_clf_idx]), 'yellow'))
                base_clf_idx += 1
                if base_clf_idx == 2:
                    base_clf_idx = 0
                statistics[stat_name](clf)


# calculateStatistics()
'''
methods(datasets)
dla każdego klasyfikatora 
wykres 
'''

# calculateStatistics()
