from __future__ import division

import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests

import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations

aucs = pd.read_csv('AUCs.txt', delimiter='\t')

aucs.columns = [u'Dataset', u'C4.5', u'C4.5+m', u'C4.5+cf', u'C4.5+m+cf']

w_stat = pd.DataFrame(columns=['Model 1', 'Model 2', 'Wilcoxon stat', 'p-value'])
k = 0
for i, j in combinations([1, 2, 3, 4], 2):
    w_stat.loc[k, 'Model 1'], w_stat.loc[k, 'Model 2'] = aucs.columns[i], aucs.columns[j]
    w_stat.loc[k, 'Wilcoxon stat'], w_stat.loc[k, 'p-value'] = stats.wilcoxon(aucs.iloc[:, i], aucs.iloc[:, j])
    k += 1

top_diff_idx = w_stat.loc[:, 'p-value'].astype(str).astype(float).nsmallest(2)

print(top_diff_idx)

diff_models_cnt = w_stat.loc[w_stat.loc[:, 'p-value'] <= 0.05, :].shape[0]
print('Number of p-value <= 0.05: %d' % diff_models_cnt)

reject_holm, p_corrected_holm, a1_holm, a2_holm = multipletests(w_stat.loc["p-value"],
                                                                alpha=0.05,
                                                                method='holm')
print("Hypothesis to reject after holm correction count: %i" % len(
    filter(lambda whether_reject: whether_reject, reject_holm)))
