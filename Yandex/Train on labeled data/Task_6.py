from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

digits = load_digits()
X = digits.data
y = digits.target
offset = int(X.shape[0] * 0.75)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]


def determine_knn(x, X, y):
    distances = ((X - x) ** 2).sum(axis=1)
    min_d = distances[0]
    min_t = y[0]
    for d, t in zip(distances, y):
        if d < min_d:
            min_d = d
            min_t = t
    return min_t


predictions = []
for i in range(X_test.shape[0]):
    x = X_test[i, :]
    predictions.append(determine_knn(x, X_train, y_train))

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

err_score = 1 - accuracy_score(y_test, knn.predict(X_test))


def write_answer_1(score):
    with open("knn.txt", "a") as file_obj:
        file_obj.write(str(score))


write_answer_1(err_score)


def write_answer_2(score):
    with open("knn2.txt", "w") as file_obj:
        file_obj.write(str(score))


forest = RandomForestClassifier(n_estimators=1000)
forest.fit(X_train, y_train)

err_score = 1 - accuracy_score(y_test, forest.predict(X_test))
write_answer_2(err_score)
