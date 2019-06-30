from __future__ import division

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.stats.weightstats import *
from statsmodels.stats.proportion import proportion_confint

import matplotlib.pyplot as plt
import seaborn as sns

water = pd.read_csv('water.txt', delimiter='\t')
print('Pearson correlation: %.4f' % stats.pearsonr(water.hardness, water.mortality)[0])
print('Spearman correlation: %.4f' % stats.spearmanr(water.hardness, water.mortality)[0])

water_south = water[water.location == 'South']
water_north = water[water.location == 'North']

a = stats.pearsonr(water_south.hardness, water_south.mortality)

print('Pearson "South" correlation: %.4f' % stats.pearsonr(water_south.hardness, water_south.mortality)[0])
print('Pearson "North" correlation: %.4f' % stats.pearsonr(water_north.hardness, water_north.mortality)[0])

bars_sex = np.array([[203., 239.], [718., 515.]])


def matthewsr(a, b, c, d):
    return (a * d - b * c) / np.sqrt((a + b) * (a + c) * (b + d) * (c + d))


matthews_coeff = matthewsr(*bars_sex.flatten())
print('Matthews correlation: %.4f' % matthews_coeff)
print('Matthews significance p-value: %f' % stats.chi2_contingency(bars_sex)[1])


def proportions_diff_confint_ind(sample1, sample2, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2.)

    p1 = sample1[0] / np.sum(sample1)
    p2 = sample2[0] / np.sum(sample2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / np.sum(sample1) + p2 * (1 - p2) / np.sum(sample2))
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / np.sum(sample1) + p2 * (1 - p2) / np.sum(sample2))

    return (left_boundary, right_boundary)


print('95%% confidence interval for a difference of men and women: [%.4f, %.4f]' %
      proportions_diff_confint_ind(bars_sex[:, 1], bars_sex[:, 0]))


def proportions_diff_z_stat_ind(sample1, sample2):
    n1 = np.sum(sample1)
    n2 = np.sum(sample2)

    p1 = sample1[0] / n1
    p2 = sample2[0] / n2
    P = float(p1 * n1 + p2 * n2) / (n1 + n2)

    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))


def proportions_diff_z_test(z_stat, alternative='two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'two-sided':
        return 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        return stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - stats.norm.cdf(z_stat)


print('p-value: %f' % proportions_diff_z_test(proportions_diff_z_stat_ind(bars_sex[:, 1], bars_sex[:, 0])))

happiness = np.array([[197., 111., 33.],
                      [382., 685., 331.],
                      [110., 342., 333.]])

print('Chi2 stat value: %.4f' % stats.chi2_contingency(happiness)[0])

st = stats.chi2_contingency(happiness)[1]

print('Chi2 stat p-value: %.62f' % stats.chi2_contingency(happiness)[1])


def cramers_stat(confusion_matrix):
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))


print('V Cramer stat value: %.4f' % cramers_stat(happiness))
