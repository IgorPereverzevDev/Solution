import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score as cv_score
import sys
from sklearn import datasets
from sklearn.datasets import fetch_olivetti_faces

from sklearn.decomposition import PCA as RandomizedPCA


# Задание 1. Автоматическое уменьшение размерности данных при помощи логарифма правдоподобия $\mathcal{L}$
# Рассмотрим набор данных размерности $D$, чья реальная размерность значительно меньше наблюдаемой (назовём её $d$). От вас требуется:
#
# Для каждого значения $\hat{d}$ в интервале [1,D] построить модель PCA с $\hat{d}$ главными компонентами.
# Оценить средний логарифм правдоподобия данных для каждой модели на генеральной совокупности, используя метод кросс-валидации с 3 фолдами (итоговая оценка значения логарифма правдоподобия усредняется по всем фолдам).
# Найти модель, для которой он максимален, и внести в файл ответа число компонент в данной модели, т.е. значение $\hat{d}_{opt}$.
# Для оценки логарифма правдоподобия модели для заданного числа главных компонент при помощи метода кросс-валидации используйте следующие функции:
#
# model = PCA(n_components=n)
# scores = cv_score(model, data)
#
# Обратите внимание, что scores -- это вектор, длина которого равна числу фолдов. Для получения оценки на правдоподобие модели его значения требуется усреднить.
#
# Для визуализации оценок можете использовать следующую функцию:
#
# plot_scores(d_scores)
#
# которой на вход передаётся вектор полученных оценок логарифма правдоподобия данных для каждого $\hat{d}$.
#
# Для интересующихся: данные для заданий 1 и 2 были сгенерированны в соответствии с предполагаемой PCA моделью. То есть: данные $Y$ с эффективной размерностью $d$, полученные из независимых равномерных распределений, линейно траснформированны случайной матрицей $W$ в пространство размерностью $D$, после чего ко всем признакам был добавлен независимый нормальный шум с дисперсией $\sigma$.1
def plot_scores(d_scores):
    n_components = np.arange(1, d_scores.size + 1)
    plt.plot(n_components, d_scores, 'b', label='PCA scores')
    plt.xlim(n_components[0], n_components[-1])
    plt.xlabel('n components')
    plt.ylabel('cv scores')
    plt.legend(loc='lower right')
    plt.show()


def write_answer_1(optimal_d):
    with open("pca_answer1.txt", "a") as fout:
        fout.write(str(optimal_d))


data = pd.read_csv('data_task1.csv')

best_score = -sys.maxsize - 1
best_components_count = 0
for i in range(1, len(data.columns)):
    model = PCA(n_components=i)
    scores = cv_score(model, data, cv=3)
    mean_score = np.mean(scores)
    if mean_score > best_score:
        best_score = mean_score
        best_components_count = i

write_answer_1(best_components_count)
print(best_components_count)
print(best_score)


# Задание 2. Ручное уменьшение размерности признаков посредством анализа дисперсии данных вдоль главных компонент
# Рассмотрим ещё один набор данных размерности $D$, чья реальная размерность значительно меньше наблюдаемой (назовём её также $d$). От вас требуется:
#
# Построить модель PCA с $D$ главными компонентами по этим данным.
# Спроецировать данные на главные компоненты.
# Оценить их дисперсию вдоль главных компонент.
# Отсортировать дисперсии в порядке убывания и получить их попарные разности: $\lambda_{(i-1)} - \lambda_{(i)}$.
# Найти разность с наибольшим значением и получить по ней оценку на эффективную размерность данных $\hat{d}$.
# Построить график дисперсий и убедиться, что полученная оценка на $\hat{d}_{opt}$ действительно имеет смысл, после этого внести полученное значение $\hat{d}_{opt}$ в файл ответа.
# Для построения модели PCA используйте функцию:
#
# model.fit(data)
#
# Для трансформации данных используйте метод:
#
# model.transform(data)
#
# Оценку дисперсий на трансформированных данных от вас потребуется реализовать вручную. Для построения графиков можно воспользоваться функцией
#
# plot_variances(d_variances)
#
# которой следует передать на вход отсортированный по убыванию вектор дисперсий вдоль компонент.2


def write_answer_2(optimal_d):
    with open("pca_answer2.txt", "w") as fout:
        fout.write(str(optimal_d))


data = pd.read_csv('data_task2.csv')

D = len(data.columns)

model = PCA(n_components=D)
model.fit(data)
data = model.transform(data)
data_transform = data.transpose()

deviations = np.std(data, axis=0)
deviations_indices = [(i, x) for i, x in enumerate(deviations)]
differences = [(deviations_indices[i][0], (deviations_indices[i - 1][1] - deviations_indices[i][1]))
               for i in range(1, D)]

# сортировка по неубыванию
answer_2 = sorted(differences, key=lambda x: -x[1])[0][0]
write_answer_2(answer_2)

# 2 Обучите метод главных компонент на датасете iris, получите преобразованные данные.
iris = datasets.load_iris()
data = iris.data
target = iris.target
target_names = iris.target_names


def plot_iris(transformed_data, target, target_names):
    plt.figure()
    for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
        plt.scatter(transformed_data[target == i, 0],
                    transformed_data[target == i, 1], c=c, label=target_name)
    plt.legend()
    plt.show()


def write_answer_3(list_pc1, list_pc2):
    with open("pca_answer3.txt", "a") as fout:
        fout.write(" ".join([str(num) for num in list_pc1]))
        fout.write(" ")
        fout.write(" ".join([str(num) for num in list_pc2]))


model = PCA(n_components=2)
model.fit(data)
iris_components = model.transform(data)

# Посчитайте корреляции исходных признаков с их проекциями на первые две главные компоненты.
# Транспонируем матрицы и находим матожидание
data_t = data.transpose()
components_t = iris_components.transpose()
data_means = [np.mean(col) for col in data_t]
components_means = [np.mean(col) for col in components_t]

# центрируем
t_means = []

data_centered = [[item - col[1] for item in col[0]] for col in zip(data_t, data_means)]
components_centered = [[item - col[1] for item in col[0]] for col in zip(components_t, components_means)]

# Для каждого признака найдите компоненту (из двух построенных), с которой он коррелирует больше всего.
first_component_features = []
second_component_features = []
max_correlations = []
for i, feature in enumerate(data_centered):
    correlations = [(j, np.correlate(feature, component)) for j, component in enumerate(components_centered)]
    max_corr = max(correlations, key=lambda c: c[1])
    if max_corr[0] == 0:
        first_component_features.append(i + 1)
    else:
        second_component_features.append(i + 1)
    max_correlations.append((i, max_corr[0], max_corr[1][0]))

plot_iris(np.array(components_centered).transpose(), target, target_names)
write_answer_3(first_component_features, second_component_features)


# 4

def write_answer_4(list_pc):
    with open("pca_answer4.txt", "a") as fout:
        fout.write(" ".join([str(num) for num in list_pc]))


data = fetch_olivetti_faces(shuffle=True, random_state=0).data
image_shape = (64, 64)

d = 10
model = RandomizedPCA(n_components=d)
model.fit(data)
faces_transformed = model.transform(data)


def center_features(matrix):
    matrix_t = matrix.transpose()
    means = [np.mean(col) for col in matrix_t]
    matrix_t_centered = [[item - col[1] for item in col[0]] for col in zip(matrix_t, means)]
    return np.array(matrix_t_centered).transpose()


data_centered = center_features(faces_transformed)


def cos_2(fi, f, component_num):
    return (fi[component_num] - f[component_num]) ** 2 / sum([(fi[k] - f[k]) ** 2 for k in range(d)])


component_faces = []
for feature in range(d):
    W_f = model.components_[feature]
    incomes_in_var = [cos_2(data_centered[i, :], W_f, feature) for i in range(data_centered.shape[0])]
    object_num_for_feature, income = max(enumerate(incomes_in_var), key=lambda x: x[1])
    component_faces.append(object_num_for_feature)

a = component_faces[0]

face = data[component_faces[0]]
plt.imshow(face.reshape(image_shape))
plt.show()

write_answer_4(component_faces)
