import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from SupervisedLearning.DecisionTrees.DecisionTreeClassifier import DecisionTreeClassifier

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
