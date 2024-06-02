import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from SupervisedLearning.DecisionTrees.DecisionTreeClassifier import DecisionTreeClassifier

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class StackingClassifier:
    def __init__(self, base_classifiers=None, meta_classifier=None):
        self.base_classifiers = base_classifiers if base_classifiers else [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)]
        self.meta_classifier = meta_classifier if meta_classifier else DecisionTreeClassifier(max_depth=1)
        self.base_predictions = None

    def fit(self, X, y):
        self.base_predictions = []
        for clf in self.base_classifiers:
            clf.fit(X, y)
            predictions = clf.predict(X)
            self.base_predictions.append(predictions)
        self.base_predictions = np.array(self.base_predictions).T
        self.meta_classifier.fit(self.base_predictions, y)

    def predict(self, X):
        base_preds = []
        for clf in self.base_classifiers:
            predictions = clf.predict(X)
            base_preds.append(predictions)
        base_preds = np.array(base_preds).T
        return self.meta_classifier.predict(base_preds)

    
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier as SklearnStackingClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers
base_classifiers = [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)]

# Train and test your implementation
your_model = StackingClassifier(base_classifiers=base_classifiers)
your_model.fit(X_train, y_train)
your_predictions = your_model.predict(X_test)
your_accuracy = accuracy_score(y_test, your_predictions)

# Train and test scikit-learn's implementation
sklearn_model = SklearnStackingClassifier(estimators=[('dt1', DecisionTreeClassifier(max_depth=1)), ('dt3', DecisionTreeClassifier(max_depth=3))],
                                           final_estimator=DecisionTreeClassifier(max_depth=1))
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

print("Your StackingClassifier Accuracy:", your_accuracy)
print("Scikit-learn StackingClassifier Accuracy:", sklearn_accuracy)
