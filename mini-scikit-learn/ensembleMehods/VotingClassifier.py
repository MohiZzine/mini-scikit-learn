from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier as SklearnVotingClassifier
import numpy as np

class VotingClassifier:
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting

    def fit(self, X, y):
        for _, estimator in self.estimators:
            estimator.fit(X, y)

    def predict(self, X):
        if self.voting == 'soft':
            predictions = np.argmax(np.mean([estimator.predict_proba(X) for _, estimator in self.estimators], axis=0), axis=1)
        else:  # 'hard' voting
            predictions = np.mean([estimator.predict(X) for _, estimator in self.estimators], axis=0)
            predictions = np.round(predictions).astype(int)
        return predictions

# Testing with Iris Dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers
base_classifiers = [
    ('logistic_regression', LogisticRegression(max_iter=1000)),
    ('decision_tree', DecisionTreeClassifier(max_depth=1)),
    ('svm', SVC(probability=True))
]

for voting in ['hard', 'soft']:
    # Custom implementation
    model = VotingClassifier(estimators=base_classifiers, voting=voting)
    model.fit(X_train, y_train)
    predictions_custom = model.predict(X_test)
    accuracy_custom = accuracy_score(y_test, predictions_custom)

    # Scikit-learn implementation
    sklearn_model = SklearnVotingClassifier(estimators=base_classifiers, voting=voting)
    sklearn_model.fit(X_train, y_train)
    predictions_sklearn = sklearn_model.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

    print(f"Custom VotingClassifier ({voting} voting) Accuracy:", accuracy_custom)
    print(f"Scikit-learn VotingClassifier ({voting} voting) Accuracy:", accuracy_sklearn)
