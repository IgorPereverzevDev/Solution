from __future__ import division

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

p = 0.75
n_stress = 67
n = 100
print("Confidence level: %.4f" % np.round(stats.binom_test(n_stress, n, p, alternative="two-sided"), 4))

# %%
n_stress = 22
n = 50
alfa = 0.05
print("Confidence interval: [%f, %f]" % proportion_confint(n_stress, n, alpha=alfa, method="wilson"))

# %%
frame = pd.read_csv("pines.txt", sep="\t", header=0)

area_size = 200.
square_count = 5
x = frame["we"].values
y = frame["sn"].values
statistic = stats.binned_statistic_2d(x, y, None, statistic="count", bins=5)
expected_mean = (float(len(x)) / (square_count ** 2))
print("Mean expected square count: %.4f" % expected_mean)

# %%
sn_num, we_num = 5, 5
trees_bins = stats.binned_statistic_2d(x, y, None, statistic='count', bins=[sn_num, we_num])
trees_squares_num = trees_bins.statistic
print(stats.chisquare(trees_squares_num.flatten(), ddof=0))

counts = statistic.statistic.reshape(-1)
low = min(counts)
high = max(counts)
trees_count = sum(counts)
expected_values = [expected_mean for x in range(0, square_count ** 2)]
chi_square = stats.chisquare(statistic.statistic.reshape(-1), expected_values, ddof=0, axis=0)
print(chi_square)
