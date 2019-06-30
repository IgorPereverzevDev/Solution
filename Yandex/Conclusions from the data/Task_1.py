from math import sqrt
import pandas as pd
import numpy as np
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic

df = pd.read_csv('water.txt', delimiter='\\t')
std = df['mortality'].std(ddof=1) / sqrt(df.mortality.count())
mean = df['mortality'].mean()
alpha = 0.95

print(
    'Mortality 95%% interval: %s' % str(_tconfint_generic(mean, std, df['mortality'].shape[0] - 1, 0.05, 'two-sided')))

water_data_south = df[df.location == 'South']
mort_mean_south = water_data_south['mortality'].mean()

mort_mean_south_std = water_data_south['mortality'].std() / np.sqrt(water_data_south['mortality'].shape[0])
print('Mortality south 95%% interval: %s' % str(_tconfint_generic(mort_mean_south, mort_mean_south_std,
                                                                  water_data_south['mortality'].shape[0] - 1,
                                                                  0.05, 'two-sided')))

from scipy import stats

print(np.ceil((stats.norm.ppf(1 - 0.05 / 2) / 0.1) ** 2))


