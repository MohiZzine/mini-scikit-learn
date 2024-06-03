import numpy as np
from collections import Counter
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from supervised_learning.decision_trees.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
        """
        Random Forest Classifier implementation.

        Parameters:
        - n_estimators (int): Number of trees in the forest.
        - max_depth (int): Maximum depth of the tree.
        - min_samples_split (int): Minimum number of samples required to split an internal node.
        - min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        - max_features ({'auto', 'sqrt', 'log2'} or int): Number of features to consider when looking for the best split.
        - bootstrap (bool): Whether bootstrap samples are used when building trees.
        - random_state (int): Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        """
        Fit the model to the given training data.

        Parameters:
        - X (array-like): Input data.
        - y (array-like): Target labels.
        """
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
        """
        Predict the class labels for the input samples.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - array: Predicted class labels.
        """
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=tree_preds)

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters:
        - X (array-like): Test samples.
        - y (array-like): True labels for X.

        Returns:
        - float: Mean accuracy of predictions.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def _bootstrap_sample(self, X, y):
        """
        Create a bootstrap sample of the dataset.

        Parameters:
        - X (array-like): Input data.
        - y (array-like): Target labels.

        Returns:
        - tuple: Bootstrap sample of X and y.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
