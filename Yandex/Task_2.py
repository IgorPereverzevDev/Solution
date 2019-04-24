import numpy
from scipy import linalg
from matplotlib import mlab
import pylab


# Сформируйте систему линейных уравнений (то есть задайте матрицу
# коэффициентов A и свободный вектор b) для многочлена первой степени,
# который должен совпадать с функцией f в точках 1 и 15.
# Решите данную систему с помощью функции scipy.linalg.solve.
# Нарисуйте функцию f и полученный многочлен. Хорошо ли он приближает исходную функцию?

def func_in(x):
    return numpy.sin(x / 5) * numpy.exp(x / 10) + 5 * numpy.exp(-x / 2)


x1 = func_in(1)
x15 = func_in(15)

M = numpy.array([[1., 1.], [1., 15.]])
V = numpy.array([x1, x15])

res = linalg.solve(M, V)


def func_out(x):
    return res[0] + res[1] * x


# Создадим список координат по оиси X на отрезке [-xmin; xmax], включая концы
x_list = mlab.frange(0, 16, 1)

# Вычислим значение функции в заданных точках
y1 = [func_in(x) for x in x_list]

# Повторите те же шаги для многочлена второй степени, который совпадает с функцией f в точках 1, 8 и 15.
# Улучшилось ли качество аппроксимации?
x1 = func_in(1)
x8 = func_in(8)
x15 = func_in(15)

M = numpy.array([[1., 1., 1.], [1., 8., 8 ** 2], [1., 15., 15 ** 2]])
V = numpy.array([x1, x8, x15])

res2 = linalg.solve(M, V)


def func_out2(x):
    return res2[0] + res2[1] * x + res2[2] * x ** 2


y2 = [func_out2(x) for x in x_list]

# Повторите те же шаги для многочлена третьей степени, который совпадает с функцией f в точках 1, 4, 10 и 15.
# Хорошо ли он аппроксимирует функцию?
# Коэффициенты данного многочлена (четыре числа в следующем порядке: w_0, w_1, w_2, w_3)
# являются ответом на задачу.Округлять коэффициенты не обязательно, но при желании можете произвести округление
# до второго знака (т.е. до числа вида 0.42)
x1 = func_in(1)
x4 = func_in(4)
x10 = func_in(10)
x15 = func_in(15)

M = numpy.array([[1., 1., 1., 1.], [1., 4., 4 ** 2, 4 ** 3], [1., 10, 10 ** 2, 10 ** 3], [1., 15., 15 ** 2, 15 ** 3]])
V = numpy.array([x1, x4, x10, x15])

res3 = linalg.solve(M, V)
res3 = [round(x, 2) for x in res3]


x_list3 = mlab.frange(0, 3, 1)


def func_out3(x):
    return numpy.sum(res3[0] + res3[1] * x + res3[2] * x ** 2 + res3 * x ** 3)


y3 = [func_out3(x) for x in x_list3]

with open('submission-2.txt', 'a') as file:
    for x in res3:
        file.write(str(x) + " ")

pylab.plot(y1)
pylab.plot(y3)
pylab.show()
