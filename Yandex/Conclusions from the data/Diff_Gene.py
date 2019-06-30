import numpy as np
import pandas as pd

from scipy import stats
from statsmodels.sandbox.stats.multicomp import multipletests

import matplotlib.pyplot as plt
import seaborn as sns

gen = pd.read_csv('gene_high_throughput_sequencing.csv')

types, cnts = np.unique(gen.Diagnosis.values, return_counts=True)
gen_normal = gen.loc[gen.Diagnosis == 'normal']
gen_neoplasia = gen.loc[gen.Diagnosis == 'early neoplasia']
gen_cancer = gen.loc[gen.Diagnosis == 'cancer']

sw_normal = gen_normal.iloc[:, 2:].apply(stats.shapiro, axis=0)
sw_normal_p = [p for _, p in sw_normal]
_, sw_normal_p_corr, _, _ = multipletests(sw_normal_p, method='fdr_bh')

sw_neoplasia = gen_neoplasia.iloc[:, 2:].apply(stats.shapiro, axis=0)
sw_neoplasia_p = [p for _, p in sw_neoplasia]
_, sw_neoplasia_p_corr, _, _ = multipletests(sw_neoplasia_p, method='fdr_bh')

sw_cancer = gen_cancer.iloc[:, 2:].apply(stats.shapiro, axis=0)
sw_cancer_p = [p for _, p in sw_cancer]
_, sw_cancer_p_corr, _, _ = multipletests(sw_cancer_p, method='fdr_bh')

print('Mean corrected p-value for "normal": %.4f' % sw_normal_p_corr.mean())
print('Mean corrected p-value for "early neoplasia": %.4f' % sw_neoplasia_p_corr.mean())
print('Mean corrected p-value for "cancer": %.4f' % sw_cancer_p_corr.mean())

tt_ind_normal_neoplasia = stats.ttest_ind(gen_normal.iloc[:, 2:], gen_neoplasia.iloc[:, 2:], equal_var=False)
tt_ind_normal_neoplasia_p = tt_ind_normal_neoplasia[1]

tt_ind_neoplasia_cancer = stats.ttest_ind(gen_neoplasia.iloc[:, 2:], gen_cancer.iloc[:, 2:], equal_var=False)
tt_ind_neoplasia_cancer_p = tt_ind_neoplasia_cancer[1]

tt_ind_normal_neoplasia_p_5 = tt_ind_normal_neoplasia_p[np.where(tt_ind_normal_neoplasia_p < 0.05)].shape[0]
tt_ind_neoplasia_cancer_p_5 = tt_ind_neoplasia_cancer_p[np.where(tt_ind_neoplasia_cancer_p < 0.05)].shape[0]

with open('answer1.txt', 'w') as fout:
    fout.write(str(tt_ind_normal_neoplasia_p_5))

with open('answer2.txt', 'w') as fout:
    fout.write(str(tt_ind_neoplasia_cancer_p_5))

# Holm correction
_, tt_ind_normal_neoplasia_p_corr, _, _ = multipletests(tt_ind_normal_neoplasia_p, method='holm')
_, tt_ind_neoplasia_cancer_p_corr, _, _ = multipletests(tt_ind_neoplasia_cancer_p, method='holm')

# Bonferroni correction
p_corr = np.array([tt_ind_normal_neoplasia_p_corr, tt_ind_neoplasia_cancer_p_corr])
_, p_corr_bonf, _, _ = multipletests(p_corr, is_sorted=True, method='bonferroni')

p_corr_bonf_normal_neoplasia_p_5 = p_corr_bonf[0][np.where(p_corr_bonf[0] < 0.05)].shape[0]
p_corr_bonf_neoplasia_cancer_p_5 = p_corr_bonf[1][np.where(p_corr_bonf[1] < 0.05)].shape[0]

print('Normal vs neoplasia samples p-values number below 0.05: %d' % p_corr_bonf_normal_neoplasia_p_5)
print('Neoplasia vs cancer samples p-values number below 0.05: %d' % p_corr_bonf_neoplasia_cancer_p_5)


def fold_change(C, T, limit=1.5):
    '''
    C - control sample
    T - treatment sample
    '''
    if T >= C:
        fc_stat = T / C
    else:
        fc_stat = -C / T

    return (np.abs(fc_stat) > limit), fc_stat


gen_p_corr_bonf_normal_p_5 = gen_normal.iloc[:, 2:].iloc[:, np.where(p_corr_bonf[0] < 0.05)[0]]
gen_p_corr_bonf_neoplasia0_p_5 = gen_neoplasia.iloc[:, 2:].iloc[:, np.where(p_corr_bonf[0] < 0.05)[0]]

fc_corr_bonf_normal_neoplasia_p_5 = 0
for norm, neopl in zip(gen_p_corr_bonf_normal_p_5.mean(), gen_p_corr_bonf_neoplasia0_p_5.mean()):
    accept, _ = fold_change(norm, neopl)
    if accept: fc_corr_bonf_normal_neoplasia_p_5 += 1

# Neoplasia vs cancer samples
gen_p_corr_bonf_neoplasia1_p_5 = gen_neoplasia.iloc[:, 2:].iloc[:, np.where(p_corr_bonf[1] < 0.05)[0]]
gen_p_corr_bonf_cancer_p_5 = gen_cancer.iloc[:, 2:].iloc[:, np.where(p_corr_bonf[1] < 0.05)[0]]

fc_corr_bonf_neoplasia_cancer_p_5 = 0
for neopl, canc in zip(gen_p_corr_bonf_neoplasia1_p_5.mean(), gen_p_corr_bonf_cancer_p_5.mean()):
    accept, _ = fold_change(neopl, canc)
    if accept: fc_corr_bonf_neoplasia_cancer_p_5 += 1

print('Normal vs neoplasia samples fold change above 1.5: %d' % fc_corr_bonf_normal_neoplasia_p_5)
print('Neoplasia vs cancer samples fold change above 1.5: %d' % fc_corr_bonf_neoplasia_cancer_p_5)

with open('answer3.txt', 'w') as fout:
    fout.write(str(fc_corr_bonf_normal_neoplasia_p_5))

with open('answer4.txt', 'w') as fout:
    fout.write(str(fc_corr_bonf_neoplasia_cancer_p_5))

# Benjamini-Hochberg correction
_, tt_ind_normal_neoplasia_p_corr, _, _ = multipletests(tt_ind_normal_neoplasia_p, method='fdr_bh')
_, tt_ind_neoplasia_cancer_p_corr, _, _ = multipletests(tt_ind_neoplasia_cancer_p, method='fdr_bh')

# Bonferroni correction
p_corr = np.array([tt_ind_normal_neoplasia_p_corr, tt_ind_neoplasia_cancer_p_corr])
_, p_corr_bonf, _, _ = multipletests(p_corr, is_sorted=True, method='bonferroni')

p_corr_bonf_normal_neoplasia_p_5 = p_corr_bonf[0][np.where(p_corr_bonf[0] < 0.05)].shape[0]
p_corr_bonf_neoplasia_cancer_p_5 = p_corr_bonf[1][np.where(p_corr_bonf[1] < 0.05)].shape[0]

print('Normal vs neoplasia samples p-values number below 0.05: %d' % p_corr_bonf_normal_neoplasia_p_5)
print('Neoplasia vs cancer samples p-values number below 0.05: %d' % p_corr_bonf_neoplasia_cancer_p_5)

# Normal vs neoplasia samples
gen_p_corr_bonf_normal_p_5 = gen_normal.iloc[:, 2:].iloc[:, np.where(p_corr_bonf[0] < 0.05)[0]]
gen_p_corr_bonf_neoplasia0_p_5 = gen_neoplasia.iloc[:, 2:].iloc[:, np.where(p_corr_bonf[0] < 0.05)[0]]

fc_corr_bonf_normal_neoplasia_p_5 = 0
for norm, neopl in zip(gen_p_corr_bonf_normal_p_5.mean(), gen_p_corr_bonf_neoplasia0_p_5.mean()):
    accept, _ = fold_change(norm, neopl)
    if accept: fc_corr_bonf_normal_neoplasia_p_5 += 1

# Neoplasia vs cancer samples
gen_p_corr_bonf_neoplasia1_p_5 = gen_neoplasia.iloc[:, 2:].iloc[:, np.where(p_corr_bonf[1] < 0.05)[0]]
gen_p_corr_bonf_cancer_p_5 = gen_cancer.iloc[:, 2:].iloc[:, np.where(p_corr_bonf[1] < 0.05)[0]]

fc_corr_bonf_neoplasia_cancer_p_5 = 0
for neopl, canc in zip(gen_p_corr_bonf_neoplasia1_p_5.mean(), gen_p_corr_bonf_cancer_p_5.mean()):
    accept, _ = fold_change(neopl, canc)
    if accept: fc_corr_bonf_neoplasia_cancer_p_5 += 1

with open('answer5.txt', 'w') as fout:
    fout.write(str(fc_corr_bonf_normal_neoplasia_p_5))

with open('answer6.txt', 'w') as fout:
    fout.write(str(fc_corr_bonf_neoplasia_cancer_p_5))
