import numpy as np
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from itertools import permutations

from tabulate import tabulate
from scipy.stats import ttest_rel
import os

plot_dir_path = './plots/'

# datasets paths
PREPARED_DATA_PATH = './prepared_datasets/'
DATASET_DIR_PATH = './datasets/'

# settings for data in datasets
MAX_NUMBER_OF_ITEMS_IN_DATASET = 20000

# settings for tests
SAMPLING_METHODS = ['random', 'least_confidence', 'margin', 'entropy']
NUMBER_OF_SAMPLING_METHODS = len(SAMPLING_METHODS)
N_SPLITS = 5
SEED = 1410
N_RUNS = 9
ALPHA = .05  # confidence threshold

def tstudent(data_for_tstudent_test, dataset_name, save_file='tstudent_results', classifier_name=''):
    t_statistic = np.zeros((NUMBER_OF_SAMPLING_METHODS, NUMBER_OF_SAMPLING_METHODS))
    p_value = np.zeros((NUMBER_OF_SAMPLING_METHODS, NUMBER_OF_SAMPLING_METHODS))
    sampling_method_number = {}
    for count, method in enumerate(SAMPLING_METHODS):
        sampling_method_number[method] = count

    all_perm = permutations(SAMPLING_METHODS, 2)
    for perm in all_perm:
        t_statistic[sampling_method_number[perm[0]], sampling_method_number[perm[1]]], p_value[
            sampling_method_number[perm[0]], sampling_method_number[perm[1]]] = ttest_rel(
            data_for_tstudent_test[perm[0]], data_for_tstudent_test[perm[1]])

    headers = SAMPLING_METHODS
    names_column = np.array([[method] for method in SAMPLING_METHODS])

    significance = np.zeros((NUMBER_OF_SAMPLING_METHODS, NUMBER_OF_SAMPLING_METHODS))
    significance[p_value >= 0.05] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)
    with open(save_file, 'a') as file:
        file.write(
            f"\n{dataset_name} {classifier_name} statistical significance "
            f"(alpha = 0.05):\n {significance_table}")


def averaged_tstudent(data_for_tstudent_test, dataset_name):
    random = [dict['random'] for dict in data_for_tstudent_test]
    margin = [dict['margin'] for dict in data_for_tstudent_test]
    least = [dict['least_confidence'] for dict in data_for_tstudent_test]
    entropy = [dict['entropy'] for dict in data_for_tstudent_test]

    data = {}
    data['random'] = [sum(i)/len(random) for i in zip(*random)]
    data['margin'] = [sum(i)/len(margin) for i in zip(*margin)]
    data['least_confidence'] = [sum(i)/len(least) for i in zip(*least)]
    data['entropy'] = [sum(i)/len(entropy) for i in zip(*entropy)]

    tstudent(data, dataset_name, save_file='tstudent_results2')



def analyze_tstudent(averaged_tstudent = True):
    if averaged_tstudent:
        save_file = 'tstudent_interpretation2'
    else:
        save_file = 'tstudent_interpretation'
    datasets = []
    with open(save_file, 'a+') as f:
        f.truncate(0)
    with open('tstudent_results2', 'r') as file:
        lines = (line.rstrip() for line in file)
        lines = list(line for line in lines if line)  # Non-blank lines in a list

        for line in lines:
            if "statistical significance" in line:
                lines_to_write = []
                line = list(line.split(" "))
                if line[0] not in datasets:
                    datasets.append(line[0])
                    with open(save_file, 'a+') as interpretation_file:
                        interpretation_file.write(f'\n##### Results for {line[0]} #####\n')
                        if not averaged_tstudent:
                            with open(save_file, 'a+') as interpretation_file:
                                interpretation_file.write(f'    Classifier {line[1]} results: \n')
                continue

            line = list(line.split())
            if any(method in line[0] for method in SAMPLING_METHODS) and len(line) == 5:
                with open(save_file, 'a+') as interpretation_file:
                    if int(line[1]) and line[0] != 'random':
                        lines_to_write.append([line[0], 'random'])
                    if int(line[2]) and line[0] != 'least_confidence':
                        lines_to_write.append([line[0], 'least_confidence'])
                    if int(line[3]) and line[0] != 'margin':
                        lines_to_write.append([line[0], 'margin'])
                    if int(line[4]) and line[0] != 'entropy':
                        lines_to_write.append([line[0], 'entropy'])
            if line[0] == 'entropy': #last line
                if len(lines_to_write):
                    with open(save_file, 'a+') as interpretation_file:
                        interpretation_file.write(' '*8)
                        interpretation_file.write('Rejecting the null hypothesis: \n')
                    lines_to_write = list(set(tuple(sorted(sub)) for sub in lines_to_write))
                    for line in lines_to_write:
                        with open(save_file, 'a+') as interpretation_file:
                            interpretation_file.write(' ' * 8)
                            interpretation_file.write(f'-{line[0]} and {line[1]} don\'t have the same values\n')
                else:
                    with open(save_file, 'a+') as interpretation_file:
                        interpretation_file.write(' ' * 8)
                        interpretation_file.write(f'Fail to reject the hypothesis that metods have the same values \n')



def classifiers_wilcoxon(scores):
    print("\nScores:\n", scores.shape)
    # Wilcoxon testing for all datasets and all sampling methods
    # scores = np.load('results.npy')
    # average of (average in every sampling method, in every split for every run)
    mean_scores = np.mean(scores, axis=(2, 3, 4)).T
    #print("\nMean scores:\n", mean_scores)

    print('\n-------------------------------')
    print('Classifiers - Wilcoxon testing:')
    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print("\nRanks:\n", ranks)

    mean_ranks = np.mean(ranks, axis=0)
    classifiers = ['gradient boost', 'logistic regression', 'random forest']
    print(classifiers)
    print("Mean ranks:\n", mean_ranks)

    from scipy.stats import ranksums

    alfa = .05  # confidence threshold
    w_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_value = np.zeros((len(classifiers), len(classifiers)))

    for i in range(len(classifiers)):
        for j in range(len(classifiers)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    from tabulate import tabulate

    headers = classifiers
    names_column = np.expand_dims(np.array(classifiers), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    # advantage matrix
    advantage = np.zeros((len(classifiers), len(classifiers)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    # check if results are statistically significant
    significance = np.zeros((len(classifiers), len(classifiers)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)


def sampling_wilcoxon(scores):
    print("\nScores:\n", scores.shape)
    print('\n-------------------------------')
    print('Sampling methods - Wilcoxon testing:')
    # average of (average in every split for every run)
    mean_scores = np.mean(scores, axis=(3, 4)).T
    print("\nMean scores:\n", mean_scores)

    ranks = []
    for ms in mean_scores:
        ranks.append(rankdata(ms).tolist())
    ranks = np.array(ranks)
    print("\nRanks:\n", ranks)

    mean_ranks = np.mean(ranks, axis=1)
    sampling_methods = ['random', 'least_confidence', 'margin', 'entropy']
    print(sampling_methods)
    print("Mean ranks:\n", mean_ranks)

    # Wilcoxon testing
    from scipy.stats import ranksums

    alfa = .05  # confidence threshold
    w_statistic = np.zeros((len(sampling_methods), len(sampling_methods)))
    p_value = np.zeros((len(sampling_methods), len(sampling_methods)))

    for i in range(len(sampling_methods)):
        for j in range(len(sampling_methods)):
            w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

    from tabulate import tabulate

    headers = sampling_methods
    names_column = np.expand_dims(np.array(sampling_methods), axis=1)
    w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
    w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

    # advantage matrix
    advantage = np.zeros((len(sampling_methods), len(sampling_methods)))
    advantage[w_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("\nAdvantage:\n", advantage_table)

    # check if results are statistically significant
    significance = np.zeros((len(sampling_methods), len(sampling_methods)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("\nStatistical significance (alpha = 0.05):\n", significance_table)


def print_num_elements(scores):
    for c_num, c in enumerate(scores):
        print('classifier', c_num + 1)
        for d_num, d in enumerate(c):
            print('dataset', d_num + 1)
            for s_num, s in enumerate(d):
                print('sampling', s_num + 1)
                for sp_num, sp in enumerate(s):
                    print('split', sp_num + 1)
                    for r_num, r in enumerate(sp):
                        print('run', r_num + 1)


def plot_accuracy(scores):
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path)

    plt.xlabel('Run number')
    plt.ylabel('Accuracy')

    datasets = ["iris","digits","wine"]
    for datset_name in os.listdir('./datasets'):
        datasets.append(os.path.splitext(datset_name)[0])

    classifiers = ['gradient boost', 'logistic regression', 'random forest']
    sampling_methods = ['random', 'least_confidence', 'margin', 'entropy']

    splits_averages = np.mean(scores, axis=3)
    print(splits_averages)
    all_results = {}
    for method in SAMPLING_METHODS:
        all_results[method] = []

    for c_num, c in enumerate(splits_averages):
        classifier_name = classifiers[c_num]
        for d_num, d in enumerate(c):
            plt.title('dataset ' + datasets[d_num] + ', ' + classifier_name)
            for s_num, s in enumerate(d):
                percentage_sampled_data = iter([15, 25, 35, 45, 55, 65, 75, 85, 95])
                x = []
                y = []
                sampling_method = sampling_methods[s_num]
                for num, accuracy in enumerate(s):
                    x.append(next(percentage_sampled_data))
                    y.append(accuracy)
                plt.plot(x, y, label=sampling_method, marker='.')
                for a, b in zip(x, y):
                    plt.text(a, b, str(round(b, 3)), alpha=.7, fontsize='x-small')
                plt.xlabel("% of classified samples")
                plt.ylabel("Avg. accuracy")
                plt.legend()
                all_results[sampling_method].append([x, y])
            plt.grid()
            plt.savefig(os.path.join(plot_dir_path, datasets[d_num] + '_' + classifier_name))
            plt.show()
            plt.clf()
    plt.title("All sampling results")
    plot_rounded_values_chart(all_results)


def plot_rounded_values_chart(all_results):
    for method in all_results:
        x = [result[0] for result in all_results[method]]
        y = [result[1] for result in all_results[method]]
        x = [sum(i)/len(x) for i in zip(*x)]
        y = [sum(i)/len(x) for i in zip(*y)]
        plt.plot(x, y, label=method, marker='.')
        for a, b in zip(x, y):
            plt.text(a, b, str(round(b, 3)), alpha=.7, fontsize='x-small')
        plt.xlabel("% of classified samples")
        plt.ylabel("Avg. accuracy")
        plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir_path, 'all_results'))
    plt.show()


if __name__ == '__main__':
    scores = np.load('results.npy')
    # classifiers_wilcoxon(scores)
    # sampling_wilcoxon(scores)
    plot_accuracy(scores)
    # t_student()

