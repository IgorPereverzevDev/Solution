import re

import numpy as np
import pylab
import scipy.spatial.distance
from matplotlib import mlab
from scipy import linalg

# SubTask 1

# Каждая строка в файле соответствует одному предложению.
# Считайте их, приведите каждую к нижнему регистру с помощью строковой функции lower().
with open('sentences.txt', 'r') as f:
    sentences = []
    words = []
    for line in f:
        line_list = re.split('[^a-z]', line.strip('\\n\\r.').lower())
        line_list = list(filter(None, line_list))
        sentences.append(line_list)
        words += line_list

# Произведите токенизацию, то есть разбиение текстов на слова.
# Для этого можно воспользоваться регулярным выражением,
# которое считает разделителем любой символ, не являющийся буквой:
# re.split('[^a-z]', t).
cat = open('sentences.txt')  # открыл файл
a = str(cat.readlines())
token_cat = re.split('[^a-z]', a)
token = [x for x in token_cat if x != 'n' and x != '']

# Составьте список всех слов, встречающихся в предложениях.
# Сопоставьте каждому слову индекс от нуля до (d - 1), где d — число различных слов в предложениях.
# Для этого удобно воспользоваться структурой dict
words = set(token)
keys = [x for x in range(len(words))]
dictionary_cats = dict(zip(keys, words))


# Создайте матрицу размера n * d, где n — число предложений.
# Заполните ее: элемент с индексом (i, j) в этой матрице должен быть равен количеству
# вхождений j-го слова в i-е предложение.
# У вас должна получиться матрица размера 22 * 254.
def file_len(f_name):
    global i
    with open(f_name) as f:
        for i, l in enumerate(f):
            pass
    return i - 1


n = file_len('sentences.txt')
d = len(words)

word_matrix = np.zeros(shape=(n, d))
for i in range(n):
    for j in range(d):
        word_matrix[i][j] = sentences[i].count(dictionary_cats[j])

# Найдите косинусное расстояние от предложения в самой первой строке
# (In comparison to dogs, cats have not undergone...)
# до всех остальных с помощью функции scipy.spatial.distance.cosine.
# Какие номера у двух предложений, ближайших к нему по этому расстоянию
# (строки нумеруются с нуля)?
# Эти два числа и будут ответами на задание.
# Само предложение (In comparison to dogs, cats have not undergone... ) имеет индекс 0.
dictOfWords = {i: scipy.spatial.distance.cosine(word_matrix[0], word_matrix[i]) for i in range(0, n)}
s = [x[0] for x in sorted(dictOfWords.items(), key=lambda x: x[1], reverse=False)]

with open('submission-1.txt', 'a') as file:
    file.write(str(s[1]) + " " + str(s[2]))


# SubTask 2

# Сформируйте систему линейных уравнений (то есть задайте матрицу
# коэффициентов A и свободный вектор b) для многочлена первой степени,
# который должен совпадать с функцией f в точках 1 и 15.
# Решите данную систему с помощью функции scipy.linalg.solve.
# Нарисуйте функцию f и полученный многочлен. Хорошо ли он приближает исходную функцию?

def func_in(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


x1 = func_in(1)
x15 = func_in(15)

M = np.array([[1., 1.], [1., 15.]])
V = np.array([x1, x15])

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

M = np.array([[1., 1., 1.], [1., 8., 8 ** 2], [1., 15., 15 ** 2]])
V = np.array([x1, x8, x15])

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

M = np.array([[1., 1., 1., 1.], [1., 4., 4 ** 2, 4 ** 3], [1., 10, 10 ** 2, 10 ** 3], [1., 15., 15 ** 2, 15 ** 3]])
V = np.array([x1, x4, x10, x15])

res3 = linalg.solve(M, V)
res3 = [round(x, 2) for x in res3]

x_list3 = mlab.frange(0, 3, 1)


def func_out3(x):
    return np.sum(res3[0] + res3[1] * x + res3[2] * x ** 2 + res3 * x ** 3)


y3 = [func_out3(x) for x in x_list3]

with open('submission-2.txt', 'a') as file:
    for x in res3:
        file.write(str(x) + " ")

pylab.plot(y1)
pylab.plot(y3)
pylab.show()
