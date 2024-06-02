import numpy as np
from collections import Counter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from SupervisedLearning.DecisionTrees.DecisionTreeClassifier import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]


# Testing the implementation
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from RandomForest import RandomForestClassifier
# Easy Test with Iris Dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train our Random Forest Classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf.predict(X_test)
accuracy = rf.score(X_test, y_test)
print(f"Custom Random Forest Classifier Accuracy on Iris dataset: {accuracy:.4f}")

# Compare with Scikit-Learn's Random Forest Classifier
sklearn_rf = SklearnRandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
sklearn_rf.fit(X_train, y_train)
y_pred_sklearn = sklearn_rf.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Scikit-Learn Random Forest Classifier Accuracy on Iris dataset: {accuracy_sklearn:.4f}")

# Medium Test with Wine Dataset
data = load_wine()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train our Random Forest Classifier
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf.predict(X_test)
accuracy = rf.score(X_test, y_test)
print(f"Custom Random Forest Classifier Accuracy on Wine dataset: {accuracy:.4f}")

# Compare with Scikit-Learn's Random Forest Classifier
sklearn_rf = SklearnRandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
sklearn_rf.fit(X_train, y_train)
y_pred_sklearn = sklearn_rf.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Scikit-Learn Random Forest Classifier Accuracy on Wine dataset: {accuracy_sklearn:.4f}")
