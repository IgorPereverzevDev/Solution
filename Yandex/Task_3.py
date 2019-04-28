# Центральная предельная теорема своими руками
#

# Выберите ваше любимое непрерывное распределение (чем меньше оно будет похоже на нормальное, тем интереснее;
# попробуйте выбрать какое-нибудь распределение из тех, что мы не обсуждали в курсе).
# Сгенерируйте из него выборку объёма 1000, постройте гистограмму выборки и нарисуйте поверх неё теоретическую плотность
# распределения вашей случайной величины.
#
#
# Для нескольких значений n (например, 5, 10, 50) сгенерируйте 1000 выборок объёма n и постройте
# гистограммы распределений их выборочных средних.
# Используя информацию о среднем и дисперсии исходного распределения (её можно без труда найти в википедии),
# посчитайте значения параметров нормальных распределений, которыми, согласно центральной предельной теореме,
# приближается распределение выборочных средних. Обратите внимание: для подсчёта значений этих параметров нужно
# использовать именно теоретические среднее и дисперсию вашей случайной величины, а не их выборочные оценки.
# Поверх каждой гистограммы нарисуйте плотность соответствующего нормального распределения.
#
# Опишите разницу между полученными распределениями при различных значениях n.
# Как меняется точность аппроксимации распределения выборочных средних нормальным с ростом n?
#
# Решение должно представлять собой IPython-ноутбук, содержащий:
# код, генерирующий выборки и графики;
# краткие описания каждого блока кода,
# объясняющие, что он делает;
# необходимые графики (убедитесь, что на них подписаны оси);
# выкладки с вычислениями параметров нормальных распределений, аппроксимирующих выборочные средние при различных n;
# выводы по результатам выполнения задания.

# Распределение Стьюдента
# Распределение Стьюдента может быть использовано для оценки того, насколько вероятно,
# что истинное среднее находится в каком-либо заданном диапазоне.

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import scipy.stats as sts

# Выбор параметров для распределения
k = 10

# Сгенерируйте из него выборку объёма 1000
sample_range = gamma.rvs(k, size=1000)

# Постройте гистограмму выборки и нарисуйте поверх неё теоретическую плотность распределения вашей случайной величины.
plt.hist(sample_range, density=True, bins=20, alpha=0.5, label='samples gamma')
plt.ylabel('number of samples')
plt.xlabel('$x$')

# теоретическая плотность распределения случайной величины
left = gamma.ppf(0.01, k)
right = gamma.ppf(0.99, k)
x = np.linspace(left, right, 1000)
plt.plot(x, gamma.pdf(x, k), 'r-', lw=5, alpha=0.7, label='erlang pdf')
plt.legend(loc='upper right')
plt.show()


# функция построения гистограммы распределений выборочных средних
# и плотности соответствующего нормального распределения
# size_samples - выбороки объёма n

def gamma_func(size_samples, Ex, Dx):
    n = size_samples
    # генерация выборок
    values = np.array([gamma.rvs(k, size=n) for x in range(1000)])
    # вычисление выборочных средних
    mean_val = values.mean(axis=1)
    plt.hist(mean_val, density=True, alpha=0.5, label='hist mean n ' + str(n))

    # мат. ожидание sigma нормального распределения
    sigma = np.math.sqrt(Dx / n)
    print('мат.ожидание=', Ex)
    print('sigma=', sigma)
    # зададим нормальное распределенние
    norm_rv = sts.norm(loc=Ex, scale=sigma)
    x = np.linspace(6, 14, 100)
    pdf = norm_rv.pdf(x)
    plt.plot(x, pdf, 'r-', lw=3, alpha=0.7, label='erlang pdf n ' + str(n))
    plt.ylabel('samples')
    plt.xlabel('$x$')
    plt.legend(loc='upper right')
    plt.show()


# Вычисление теоритических EX, std, DX  распределения
EX = gamma.mean(k)
std = gamma.std(k)
DX = std ** 2
print('Ex=', EX, ' STD=', std, ' DX=', DX)

gamma_func(5, EX, DX)
gamma_func(10, EX, DX)
gamma_func(50, EX, DX)

# ## Вывод:
# Распределение выборочных средних для функции gamma хорошо описывается нормальным распределением.
# С ростом n точность аппроксимации увеличивается.
