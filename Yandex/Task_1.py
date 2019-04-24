import re

import numpy as np
import scipy.spatial.distance

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
    file.write(str(s[1])+" " + str(s[2]))
