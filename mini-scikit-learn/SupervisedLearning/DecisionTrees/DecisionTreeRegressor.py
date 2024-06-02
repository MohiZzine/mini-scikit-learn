import numpy as np

class DecisionTreeRegressor:
    def __init__(self, criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.tree_ = None

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean((y - predictions) ** 2)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if (self.max_depth and depth >= self.max_depth) or num_samples < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = self._mean_of_values(y)
            return Node(value=leaf_value)

        best_feat, best_thresh = self._best_split(X, y, num_features)
        if best_feat is None:
            leaf_value = self._mean_of_values(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, num_features):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in range(num_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._variance_reduction(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _variance_reduction(self, y, X_column, split_thresh):
        parent_variance = np.var(y)

        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        num_samples = len(y)
        num_left, num_right = len(left_idxs), len(right_idxs)
        left_variance, right_variance = np.var(y[left_idxs]), np.var(y[right_idxs])
        weighted_variance = (num_left / num_samples) * left_variance + (num_right / num_samples) * right_variance

        reduction = parent_variance - weighted_variance
        return reduction

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _mean_of_values(self, y):
        return np.mean(y)

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Testing the DecisionTreeRegressor

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor

def test():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train our Decision Tree Regressor
    dt = DecisionTreeRegressor(max_depth=3)
    dt.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred_train = dt.predict(X_train)
    y_pred_test = dt.predict(X_test)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"Custom Decision Tree Regressor Train MSE: {mse_train:.4f}")
    print(f"Custom Decision Tree Regressor Test MSE: {mse_test:.4f}")

    # Compare with Scikit-Learn's Decision Tree Regressor
    sklearn_dt = SklearnDecisionTreeRegressor(max_depth=3)
    sklearn_dt.fit(X_train, y_train)
    y_pred_sklearn_train = sklearn_dt.predict(X_train)
    y_pred_sklearn_test = sklearn_dt.predict(X_test)
    mse_sklearn_train = mean_squared_error(y_train, y_pred_sklearn_train)
    mse_sklearn_test = mean_squared_error(y_test, y_pred_sklearn_test)
    print(f"Scikit-Learn Decision Tree Regressor Train MSE: {mse_sklearn_train:.4f}")
    print(f"Scikit-Learn Decision Tree Regressor Test MSE: {mse_sklearn_test:.4f}")

if __name__ == "__main__":
    test()
