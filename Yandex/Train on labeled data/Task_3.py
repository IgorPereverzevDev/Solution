import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

data = datasets.load_boston()

X = data['data']
Y = data['target']

part = int(X.shape[0] * 0.75)
X_train, Y_train = X[:part], Y[:part]
X_test, Y_test = X[part:], Y[part:]


def gradient(y, p):
    p = p.reshape(-1)
    return 2.0 * (y - p > 0.0) - 1.0


def gbm_predict(X):
    return [sum([coeff * algo.predict([x])[0] for algo, coeff in
                 zip(base_algorithms_list, coefficients_list)]) for x in X]


base_algorithms_list = []
coefficients_list = []

scores = []
trees = np.arange(5, 1000, 100)
for i in trees:
    estimator = GradientBoostingRegressor(n_estimators=i, learning_rate=0.01, max_depth=4, random_state=42, loss='ls')
    estimator.fit(X_train, Y_train)
    rmse = mean_squared_error(Y_test, gbm_predict(X_test)) ** 0.5
    scores.append(rmse)

plt.plot(trees, scores)
plt.xlabel('trees')
plt.ylabel('scores')
plt.grid(True)
plt.show()

tree_depth = np.arange(1, 50, 10)
scoring = []
for max_depth in tree_depth:
    est = GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, max_depth=max_depth, random_state=42,
                                    loss='ls')
    est.fit(X_train, Y_train)
    rmse = mean_squared_error(Y_test, est.predict(X_test)) ** 0.5
    scoring.append(rmse)


def write_answer_5(rmse):
    with open("grad_boosting-5.txt", "w") as file_obj:
        file_obj.write(str(rmse))


reg = LinearRegression()
reg.fit(X_train, Y_train)

rmse = mean_squared_error(Y_test, reg.predict(X_test)) ** 0.5

write_answer_5(rmse)
