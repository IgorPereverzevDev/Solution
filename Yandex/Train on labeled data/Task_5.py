import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.datasets import load_digits

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

breast_cancer_scores = []
estimator = BernoulliNB()
estimator.fit(X, y)

score = np.mean(cross_val_score(estimator, X, y, n_jobs=-1))
breast_cancer_scores.append(score)
print(score)

estimator = MultinomialNB()
estimator.fit(X, y)

score = np.mean(cross_val_score(estimator, X, y, n_jobs=-1))
breast_cancer_scores.append(score)
print(score)

estimator = GaussianNB()
estimator.fit(X, y)

score = np.mean(cross_val_score(estimator, X, y, n_jobs=-1))
breast_cancer_scores.append(score)
print(score)

digits = load_digits()
X, y = digits.data, digits.target

digits_scores = []

estimator = BernoulliNB()
estimator.fit(X, y)

score = np.mean(cross_val_score(estimator, X, y, n_jobs=-1))
digits_scores.append(score)
print(score)

estimator = MultinomialNB()
estimator.fit(X, y)

score = np.mean(cross_val_score(estimator, X, y, n_jobs=-1))
digits_scores.append(score)
print(score)

estimator = GaussianNB()
estimator.fit(X, y)

score = np.mean(cross_val_score(estimator, X, y, n_jobs=-1))
digits_scores.append(score)
print(score)


def write_answer_1(score):
    with open("first.txt", "a") as file_obj:
        file_obj.write(str(score))


def write_answer_2(score):
    with open("second.txt", "a") as file_obj:
        file_obj.write(str(score))


def write_answer_3():
    with open("third.txt", "a") as file_obj:
        file_obj.write(str(3) + " " + str(4))


write_answer_1(max(breast_cancer_scores))
write_answer_2(max(digits_scores))
write_answer_3()
