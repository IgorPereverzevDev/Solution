from __future__ import division

import numpy as np
import pandas as pd

from scipy import stats

surv_data = np.array([49, 58, 75, 110, 112, 132, 151, 276, 281, 362])
exp_surv = 200
print(stats.wilcoxon(surv_data - exp_surv))

forest_not_cut = np.array([22, 22, 15, 13, 19, 19, 18, 20, 21, 13, 13, 15])

forest_cut = np.array([17, 18, 18, 15, 12, 4, 14, 15, 10])

print(stats.mannwhitneyu(forest_not_cut, forest_cut, alternative='greater'))

challenger = pd.read_csv('challenger.txt', delimiter='\t')


def get_bootstrap_samples(data, n_samples):
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples


def stat_intervals(stat, alpha):
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries


challenger_broken = challenger[challenger['Incident'] == 1]['Temperature'].values
challenger_not_broken = challenger[challenger['Incident'] == 0]['Temperature'].values

np.random.seed(0)
challenger_broken_bs_mean = np.array(list(map(np.mean, get_bootstrap_samples(challenger_broken, 1000))))
challenger_not_broken_bs_mean = np.array(list(
    map(np.mean, get_bootstrap_samples(challenger_not_broken, 1000))))

print('95%% confidence interval for times decrease of infarction: %s' %
      str(stat_intervals(challenger_broken_bs_mean - challenger_not_broken_bs_mean, 0.05)))


def get_random_combinations(n1, n2, max_combinations):
    index = list(range(n1 + n2))
    indices = {tuple(index)}
    for i in range(max_combinations - 1):
        np.random.shuffle(index)
        indices.add(tuple(index))
    return [(index[:n1], index[n1:]) for index in indices]


def permutation_t_stat_ind(sample1, sample2):
    return np.mean(sample1) - np.mean(sample2)


def permutation_zero_dist_ind(sample1, sample2, max_combinations=None):
    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)

    if max_combinations:
        indices = get_random_combinations(n1, len(sample2), max_combinations)
    else:
        indices = [(list(index), filter(lambda i: i not in index, range(n))) \
                   for index in np.itertools.combinations(range(n), n1)]

    distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() \
             for i in indices]
    return distr


def permutation_test(sample, mean, max_permutations=None, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    t_stat = permutation_t_stat_ind(sample, mean)

    zero_distr = permutation_zero_dist_ind(sample, mean, max_permutations)

    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)


np.random.seed(0)
print('p-value: %.4f' % permutation_test(challenger_broken, challenger_not_broken, max_permutations=10000))
