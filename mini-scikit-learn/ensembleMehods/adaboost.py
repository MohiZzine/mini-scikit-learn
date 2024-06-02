import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
from SupervisedLearning.DecisionTrees.DecisionTreeClassifier import DecisionTreeClassifier

sys.path.append(os.path.dirname(os.path.dirname(__file__)))



class AdaBoostClassifier:
    def __init__(self, base_estimator=None,
                 n_estimators=50, learning_rate=1.0):
        self.base_estimator = (
            base_estimator if base_estimator
            else DecisionTreeClassifier(max_depth=1)
        )
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []
        self.estimator_weights_ = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        sample_weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            estimator = self.base_estimator
            estimator.fit(X, y, sample_weight=sample_weights)
            y_pred = estimator.predict(X)

            incorrect = (y_pred != y)
            estimator_error = np.mean(
                np.average(incorrect, weights=sample_weights, axis=0)
            )

            if estimator_error > 0.5:
                break

            estimator_weight = self.learning_rate * np.log(
                (1 - estimator_error) / estimator_error
            )
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)

            sample_weights *= np.exp(estimator_weight * (incorrect != 0))
            sample_weights /= np.sum(sample_weights)

    def predict(self, X):
        final_predictions = np.zeros(X.shape[0])

        for estimator, weight in zip(self.estimators_,
                                     self.estimator_weights_):
            final_predictions += weight * estimator.predict(X)

        return np.sign(final_predictions)


# Testing with Iris Dataset
iris = load_iris()
X, y = iris.data, iris.target
# AdaBoost typically uses -1 and 1 for binary classification
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple AdaBoost Classifier
adaboost_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50
)
adaboost_clf.fit(X_train, y_train)
y_pred = adaboost_clf.predict(X_test)

print(
    "Accuracy of Simple AdaBoost Classifier on Iris dataset:",
    accuracy_score(y_test, y_pred)
)

# Testing with Breast Cancer Dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
y = np.where(y == 0, -1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple AdaBoost Classifier
adaboost_clf = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50
)
adaboost_clf.fit(X_train, y_train)
y_pred = adaboost_clf.predict(X_test)

print(
    "Accuracy of Simple AdaBoost Classifier on Breast Cancer dataset:",
    accuracy_score(y_test, y_pred)
)
