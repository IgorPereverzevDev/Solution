import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
from statsmodels.stats.proportion import proportion_confint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data_exp = np.array([1 if i < 10 else 0 for i in range(34)])
data_ctrl = np.array([1 if i < 4 else 0 for i in range(16)])

conf_interval_banner_exp = proportion_confint(np.sum(data_exp), len(data_exp), method='wilson')
conf_interval_banner_ctrl = proportion_confint(np.sum(data_ctrl), len(data_ctrl), method='wilson')


def proportions_diff_confint_ind(sample1, sample2, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2.)

    p1 = float(sum(sample1)) / len(sample1)
    p2 = float(sum(sample2)) / len(sample2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / len(sample1) + p2 * (1 - p2) / len(sample2))
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / len(sample1) + p2 * (1 - p2) / len(sample2))

    return (left_boundary, right_boundary)


def proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)

    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2
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


print('p-value: %.4f' % proportions_diff_z_test(proportions_diff_z_stat_ind(data_exp, data_ctrl), 'greater'))

frame = pd.read_csv("banknotes.txt", sep="\t", header=0)
banknotes_train, banknotes_test = train_test_split(frame, test_size=50, random_state=1)

features_1 = [u'X1', u'X2', u'X3']
clf_logreg_1 = LogisticRegression()
clf_logreg_1.fit(banknotes_train[features_1].values, banknotes_train[u'real'].values)

pred_1 = clf_logreg_1.predict(banknotes_test[features_1].values)
print('Error rate pred1: %f' % (1 - accuracy_score(banknotes_test[u'real'].values, pred_1)))
err_1 = np.array([1 if banknotes_test[u'real'].values[i] == pred_1[i] else 0 for i in range(len(pred_1))])

features_2 = [u'X4', u'X5', u'X6']
clf_logreg_2 = LogisticRegression()
clf_logreg_2.fit(banknotes_train[features_2].values, banknotes_train[u'real'].values)
pred_2 = clf_logreg_2.predict(banknotes_test[features_2].values)
print('Error rate pred2: %f' % (1 - accuracy_score(banknotes_test[u'real'].values, pred_2)))
err_2 = np.array([1 if banknotes_test[u'real'].values[i] == pred_2[i] else 0 for i in range(len(pred_2))])


def proportions_diff_confint_rel(sample1, sample2, alpha=0.05):
    z = stats.norm.ppf(1 - alpha / 2.)
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    left_boundary = float(f - g) / n - z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    right_boundary = float(f - g) / n + z * np.sqrt(float((f + g)) / n ** 2 - float((f - g) ** 2) / n ** 3)
    return (left_boundary, right_boundary)


print(
    '95%% confidence interval for a difference between predictions: [%.4f, %.4f]' % proportions_diff_confint_rel(err_1,
                                                                                                                 err_2))


def proportions_diff_z_stat_rel(sample1, sample2):
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([1 if (x[0] == 1 and x[1] == 0) else 0 for x in sample])
    g = sum([1 if (x[0] == 0 and x[1] == 1) else 0 for x in sample])

    return float(f - g) / np.sqrt(f + g - float((f - g) ** 2) / n)


print('p-value: %f' % proportions_diff_z_test(proportions_diff_z_stat_rel(err_1, err_2)))

n = 200000.
miu = 525.
std = 100.
n_yr = 100.
mean_yr = 541.4

# Проверьте гипотезу о неэффективности программы против односторонней альтернативы о том, что программа работает.
# Отвергается ли на уровне значимости 0.05 нулевая гипотеза? Введите достигаемый уровень значимости, округлённый до 4 знаков после десятичной точки.
# %%
from scipy.stats import norm


def check_hypothesis(mean, n, miu, std):
    Z = (mean - miu) / (std / np.sqrt(n))
    return (Z, 1. - norm.cdf(Z))


# %%
print(np.round(check_hypothesis(mean_yr, n_yr, miu, std), 4))

# Оцените теперь эффективность подготовительных курсов, средний балл 100 выпускников которых равен 541.5.
# Отвергается ли на уровне значимости 0.05 та же самая нулевая гипотеза против той же самой альтернативы?
# Введите достигаемый уровень значимости, округлённый до 4 знаков после десятичной точки.
# %%
mean_yr = 541.5
print(np.round(check_hypothesis(mean_yr, n_yr, miu, std), 4))

