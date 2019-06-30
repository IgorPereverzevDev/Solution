import numpy as np
import pandas as pd

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

import matplotlib.pyplot as plt
import seaborn as sns

botswana = pd.read_csv("botswana.tsv", sep="\t", header=0)
botswana.head()
# О каждой из них мы знаем:
#    сколько детей она родила (признак ceb)
#    возраст (age)
#    длительность получения образования (educ)
#    религиозная принадлежность (religion)
#    идеальное, по её мнению, количество детей в семье (idlnchld)
#    была ли она когда-нибудь замужем (evermarr)
#    возраст первого замужества (agefm)
#    длительность получения образования мужем (heduc)
#    знает ли она о методах контрацепции (knowmeth)
#    использует ли она методы контрацепции (usemeth)
#    живёт ли она в городе (urban)
#    есть ли у неё электричество, радио, телевизор и велосипед (electric, radio, tv, bicycle)
# Давайте научимся оценивать количество детей ceb по остальным признакам.
# Сколько разных значений принимает признак religion?
# %%
print("Religion takes %i different values" % len(set(botswana.religion.values)))

# Во многих признаках есть пропущенные значения. Сколько объектов из 4361 останется, если выбросить все, содержащие пропуски?
# %%
print("Fully filled objects count: %i" % botswana.dropna(axis=0, how="any").shape[0])

# В разных признаках пропуски возникают по разным причинам и должны обрабатываться по-разному.
# Например, в признаке agefm пропуски стоят только там, где evermarr=0, то есть, они соответствуют женщинам, никогда не выходившим замуж.
# Таким образом, для этого признака NaN соответствует значению "не применимо".
# В подобных случаях, когда признак x1 на части объектов в принципе не может принимать никакие значения, рекомендуется поступать так:
#   создать новый бинарный признак
#       x2=1, если x1='не применимо', иначе x2=0;
#   заменить "не применимо" в x1 на произвольную константу c, которая среди других значений x1 не встречается.
# Теперь, когда мы построим регрессию на оба признака и получим модель вида
#       y=β0+β1x1+β2x2,
# на тех объектах, где x1 было измерено, регрессионное уравнение примет вид
#       y=β0+β1x,
# а там, где x1 было "не применимо", получится
#       y=β0+β1c+β2.
# Выбор c влияет только на значение и интерпретацию β2, но не β1.
# Давайте используем этот метод для обработки пропусков в agefm и heduc.
#   Создайте признак nevermarr, равный единице там, где в agefm пропуски.
#   Удалите признак evermarr — в сумме с nevermarr он даёт константу, значит, в нашей матрице X будет мультиколлинеарность.
#   Замените NaN в признаке agefm на cagefm=0.
#   У объектов, где nevermarr = 1, замените NaN в признаке heduc на cheduc1=−1 (ноль использовать нельзя, так как он уже встречается у некоторых объектов выборки).
# Сколько осталось пропущенных значений в признаке heduc?


botswana['nevermarr'] = [1 if botswana.loc[i, 'evermarr'] == 0 else 0 for i in range(botswana.shape[0])]
np.unique(botswana.agefm[botswana.agefm.notnull()].values)
del botswana['evermarr']
botswana.heduc[botswana.heduc.isnull() & botswana.nevermarr.values == 1] = -1

print(botswana.heduc.isnull().value_counts())

botswana['idlnchld_noans'] = 0
botswana.loc[botswana.idlnchld.isnull(), 'idlnchld_noans'] = 1

botswana['heduc_noans'] = 0
botswana.loc[botswana.heduc.isnull(), 'heduc_noans'] = 1

botswana['usemeth_noans'] = 0
botswana.loc[botswana.usemeth.isnull(), 'usemeth_noans'] = 1

botswana.idlnchld[botswana.idlnchld.isnull()] = -1
botswana.heduc[botswana.heduc.isnull()] = -2
botswana.usemeth[botswana.usemeth.isnull()] = -1
botswana = botswana.dropna()

elem_num = botswana.shape[0] * botswana.shape[1]
print('Array size: %d' % elem_num)

botswana.info()
formula = 'ceb ~ ' + ' + '.join(botswana.columns[1:])

reg_m = smf.ols(formula, data=botswana)
fitted_m = reg_m.fit()
print(fitted_m.summary())

print(botswana.religion.value_counts())
print('Breusch-Pagan test: p=%f' % sms.het_breuschpagan(fitted_m.resid, fitted_m.model.exog)[1])

reg_m2 = smf.ols(formula, data=botswana)
fitted_m2 = reg_m2.fit(cov_type='HC1')
print(fitted_m2.summary())

formula2 = 'ceb ~ age + educ + idlnchld + knowmeth + usemeth + agefm + heduc + urban + electric + bicycle \
+ nevermarr + idlnchld_noans + heduc_noans + usemeth_noans'

reg_m3 = smf.ols(formula2, data=botswana)
fitted_m3 = reg_m3.fit()
print(fitted_m3.summary())

print('Breusch-Pagan test: p=%f' % sms.het_breuschpagan(fitted_m3.resid, fitted_m3.model.exog)[1])
reg_m4 = smf.ols(formula2, data=botswana)
fitted_m4 = reg_m4.fit(cov_type='HC1')
print(fitted_m4.summary())

p=0.0000000000000000000000000000000000000003

print('F=%f, p=%f, k1=%f' % reg_m2.fit().compare_f_test(reg_m4.fit()))
