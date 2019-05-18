import matplotlib.pyplot as plt
import numpy as np
from pybrain.utilities import percentError

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

from pybrain.datasets import ClassificationDataSet  # Структура данных pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer

with open('winequality-red.csv') as f:
    f.readline()  # пропуск заголовочной строки
    data = np.loadtxt(f, delimiter=';')

TRAIN_SIZE = 0.7  # Разделение данных на обучающую и контрольную части в пропорции 70/30%

y = data[:, -1]
np.place(y, y < 5, 5)
np.place(y, y > 7, 7)
y -= min(y)
X = data[:, :-1]
X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=0)

# Конвертация данных в структуру ClassificationDataSet
# Обучающая часть
ds_train = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
# Первый аргумент -- количество признаков np.shape(X)[1], второй аргумент -- количество меток классов len(np.unique(y_train)))
ds_train.setField('input', X_train)  # Инициализация объектов
ds_train.setField('target', y_train[:, np.newaxis])  # Инициализация ответов; np.newaxis создает вектор-столбец
ds_train._convertToOneOfMany()  # Бинаризация вектора ответов
# Контрольная часть
ds_test = ClassificationDataSet(np.shape(X)[1], nb_classes=len(np.unique(y_train)))
ds_test.setField('input', X_test)
ds_test.setField('target', y_test[:, np.newaxis])
ds_test._convertToOneOfMany()

np.random.seed(0)  # Зафиксируем seed для получния воспроизводимого результата


def plot_classification_error(hidden_neurons_num, res_train_vec, res_test_vec):
    plt.figure()
    plt.plot(hidden_neurons_num, res_train_vec)
    plt.plot(hidden_neurons_num, res_test_vec, '-r')


def write_answer_nn(optimal_neurons_num):
    with open("nnets_answer2.txt", "w") as fout:
        fout.write(str(optimal_neurons_num))


hidden_neurons_num = [50, 100, 200, 500, 700, 1000]
res_train_vec = list()
res_test_vec = list()
# Определение основных констант
HIDDEN_NEURONS_NUM = 70  # Количество нейронов, содержащееся в скрытом слое сети
MAX_EPOCHS = 150  # Максимальное число итераций алгоритма оптимизации параметров сети

for nnum in hidden_neurons_num:
    net = buildNetwork(ds_train.indim, nnum, ds_train.outdim, outclass=SoftmaxLayer)
    net._setParameters(np.random.random((len(net.params))))

    trainer = BackpropTrainer(net, dataset=ds_train)
    err_train, err_val = trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)

    res_train = net.activateOnDataset(ds_train).argmax(axis=1)
    res_train_vec.append(percentError(res_train, ds_train['target'].argmax(axis=1)))

    res_test = net.activateOnDataset(ds_test).argmax(axis=1)
    res_test_vec.append(percentError(res_test, ds_test['target'].argmax(axis=1)))
# Постройте график зависимости ошибок на обучении и контроле в зависимости от количества нейронов
# plot_classification_error(hidden_neurons_num, res_train_vec, res_test_vec)
#  Запишите в файл количество нейронов, при котором достигается минимум ошибки на контроле
write_answer_nn(hidden_neurons_num[np.argmin(res_test_vec)])
