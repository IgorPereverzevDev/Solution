import numpy as np
from scipy.stats import norm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

avg = 9.5
std = 0.4
n = 160
avg_sel = 9.57
Z = (avg_sel - avg) / (std / np.sqrt(n))
p_val = 2. * (1. - norm.cdf(Z))
print(np.round((Z, p_val, norm.cdf(Z)), 4))

# Имеются данные о стоимости и размерах 53940 бриллиантов.
# %%


frame = pd.read_csv("diamonds.txt", sep="\t", header=0)
frame.head()

# Отделите 25% случайных наблюдений в тестовую выборку с помощью функции
# sklearn.cross_validation.train_test_split (зафиксируйте random state = 1).
# %%


y = frame["price"].values
x = frame.drop("price", axis=1).values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

# На обучающей выборке настройте две регрессионные модели:
#     1. линейную регрессию с помощью LinearRegression без параметров
#     2. случайный лес с помощью RandomForestRegressor с random_state=1.
# %%


linear = LinearRegression().fit(x_train, y_train)
forest = RandomForestRegressor(random_state=1).fit(x_train, y_train)

# Какая из моделей лучше предсказывает цену бриллиантов?
# Сделайте предсказания на тестовой выборке.
# %%

forest_predictions = map(lambda x_row: forest.predict(x_row.reshape(1, -1))[0], x_test)
linear_predictions = map(lambda x_row: linear.predict(x_row.reshape(1, -1))[0], x_test)

# Посчитайте модули отклонений предсказаний от истинных цен.
# %%
forest_abs_devs = np.array(list(
    map(lambda prediction_y_val: np.abs(prediction_y_val[0] - prediction_y_val[1]), zip(forest_predictions, y_test))))
linear_abs_devs = np.array(list(
    map(lambda prediction_y_val: np.abs(prediction_y_val[0] - prediction_y_val[1]), zip(linear_predictions, y_test))))

# Проверьте гипотезу об одинаковом среднем качестве предсказаний.
# %%
from statsmodels.stats.weightstats import *

# %%
statistic, p = stats.shapiro(forest_abs_devs - linear_abs_devs)

stats.ttest_rel(abs(y_test - linear.predict(x_test)),
                abs(y_test - forest.predict(x_test)))

print("Shapiro-Wilk normality test, W-statistic: {}, p-value: {}".format(statistic, p))
# Вычислите достигаемый уровень значимости.
# Отвергается ли гипотеза об одинаковом качестве моделей против двусторонней альтернативы на уровне значимости
# α=0.05?
# %%
stats.ttest_rel(linear_abs_devs, forest_abs_devs)

# В предыдущей задаче посчитайте 95% доверительный интервал для разности средних абсолютных ошибок предсказаний
# регрессии и случайного леса. Чему равна его ближайшая к нулю граница? Округлите до десятков (поскольку
# случайный лес может давать немного разные предсказания в зависимости от версий библиотек, мы просим вас так
# сильно округлить, чтобы полученное значение наверняка совпало с нашим).
# %%
print("95%% confidence interval: [%f, %f]" % DescrStatsW(linear_abs_devs - forest_abs_devs).tconfint_mean())
