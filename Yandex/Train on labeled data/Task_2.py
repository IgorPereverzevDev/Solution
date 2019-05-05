import pandas as pd
import numpy as np
import seaborn as sns


def write_answer_to_file(answer, filename):
    with open(filename, 'w') as f_out:
        f_out.write(str(round(answer, 3)))


adver_data = pd.read_csv('advertising.csv')

adver_data.head(5)

sns.pairplot(adver_data)

X = adver_data[['TV', 'Radio', 'Newspaper']]
y = adver_data.Sales

means = X.apply(np.mean)
stds = X.apply(np.std)

X = X.apply(lambda x: (x - means) / stds, axis=1)

X.apply(np.mean)

X.apply(np.std)

X['x0'] = 1
X = X[['x0', 'TV', 'Radio', 'Newspaper']]


def mserror(y, y_pred):
    return sum(map(lambda x1, x2: (x1 - x2) ** 2, y, y_pred)) / len(y)


y_mean_sales = [np.median(y)] * len(y)
answer1 = mserror(y, y_mean_sales)
print(answer1)
write_answer_to_file(answer1, '1.txt')


def normal_equation(X, y):
    A = np.dot(np.transpose(X), X)
    b = np.dot(np.transpose(X), y)
    return np.linalg.solve(A, b)


norm_eq_weights = normal_equation(X, y)
print(norm_eq_weights)


def predict(data, weights):
    return np.dot([1] + data, weights)


answer2 = predict([0, 0, 0], norm_eq_weights)
print(answer2)
write_answer_to_file(answer2, '2.txt')


def linear_prediction(X, w):
    return X.apply(lambda x: np.dot(x, w), axis=1)


answer3 = mserror(y, linear_prediction(X, norm_eq_weights))
print(answer3)
write_answer_to_file(answer3, '3.txt')


def stochastic_gradient_step(X, y, w, train_ind, eta=0.01):
    l = len(y)
    x_k = X.values[train_ind]
    y_k = y.values[train_ind]
    return w + 2 * eta / l * x_k * (y_k - np.dot(w, x_k))


def stochastic_gradient_descent(X, y, w_init, eta=1e-2, max_iter=1e4,
                                min_weight_dist=1e-8, seed=42):
    # Инициализируем расстояние между векторами весов на соседних
    # итерациях большим числом. 
    weight_dist = np.inf
    # Инициализируем вектор весов
    w = w_init
    # Сюда будем записывать ошибки на каждой итерации
    errors = []
    # Счетчик итераций
    iter_num = 0
    # Будем порождать псевдослучайные числа 
    # (номер объекта, который будет менять веса), а для воспроизводимости
    # этой последовательности псевдослучайных чисел используем seed.
    np.random.seed(seed)

    # Основной цикл
    while weight_dist > min_weight_dist and iter_num < max_iter:
        # порождаем псевдослучайный 
        # индекс объекта обучающей выборки
        random_ind = np.random.randint(X.shape[0])

        w_next = stochastic_gradient_step(X, y, w, random_ind, eta)
        y_pred = linear_prediction(X, w_next)
        errors.append(mserror(y, y_pred))

        weight_dist = np.linalg.norm(w - w_next)

        iter_num += 1
        w = w_next

    return w, errors


stoch_grad_desc_weights, stoch_errors_by_iter = stochastic_gradient_descent(X, y, np.zeros(4), max_iter=1e5)

answer4 = stoch_errors_by_iter[-1]
print(answer4)
write_answer_to_file(answer4, '4.txt')
