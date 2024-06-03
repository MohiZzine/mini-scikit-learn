import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from supervised_learning.decision_trees.decision_tree_regressor import DecisionTreeRegressor

class RandomForestRegressor:
    def __init__(self, n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                 verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
        """
        Random Forest Regressor implementation.

        Parameters:
        - n_estimators (int): Number of trees in the forest.
        - criterion ({'squared_error'}): The function to measure the quality of a split.
        - max_depth (int): Maximum depth of the tree.
        - min_samples_split (int): Minimum number of samples required to split an internal node.
        - min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
        - min_weight_fraction_leaf (float): Minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
        - max_features (int): Number of features to consider when looking for the best split.
        - max_leaf_nodes (int): Grow trees with max_leaf_nodes in best-first fashion.
        - min_impurity_decrease (float): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        - bootstrap (bool): Whether bootstrap samples are used when building trees.
        - oob_score (bool): Whether to use out-of-bag samples to estimate the R^2 on unseen data.
        - n_jobs (int): The number of jobs to run in parallel for both fit and predict. -1 means using all processors.
        - random_state (int): Random seed for reproducibility.
        - verbose (int): Controls the verbosity when fitting and predicting.
        - warm_start (bool): When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.
        - ccp_alpha (float): Complexity parameter used for Minimal Cost-Complexity Pruning.
        - max_samples (int or float): If bootstrap is True, the number of samples to draw from X to train each base estimator.
                                      If None, then draw X.shape[0] samples.
        - monotonic_cst (list): List of bool. If set to True, the feature should have a positive or negative effect on the prediction.

        Attributes:
        - estimators_ (list): The collection of fitted sub-estimators.
        - estimators_samples_ (list): The subset of drawn samples (indices) used for fitting each estimator.
        - oob_prediction_ (array): Prediction computed with out-of-bag estimate on the training set.
        - oob_score_ (float): Score of the training dataset obtained using an out-of-bag estimate.
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.monotonic_cst = monotonic_cst
        self.estimators_ = []
        self.estimators_samples_ = []
        self.oob_prediction_ = None
        self.oob_score_ = None

    def fit(self, X, y):
        """
        Fit the model to the given training data.

        Parameters:
        - X (array-like): Input data.
        - y (array-like): Target labels.

        Returns:
        - self: Returns an instance of self.
        """
        n_samples, n_features = X.shape
        self.estimators_ = []
        self.estimators_samples_ = []
        random_state = np.random.RandomState(self.random_state)

        for _ in range(self.n_estimators):
            estimator = DecisionTreeRegressor(
                criterion=self.criterion, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                random_state=random_state.randint(np.iinfo(np.int32).max)
            )
            
            if self.bootstrap:
                if self.max_samples:
                    if isinstance(self.max_samples, float):
                        n_bootstrap_samples = int(self.max_samples * n_samples)
                    else:
                        n_bootstrap_samples = self.max_samples
                else:
                    n_bootstrap_samples = n_samples
                
                indices = random_state.choice(n_samples, n_bootstrap_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            
            X_sample = X[indices]
            y_sample = y[indices]
            
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)
            self.estimators_samples_.append(indices)

        if self.oob_score:
            self._set_oob_score(X, y)
        
        return self

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters:
        - X (array-like): Input data.

        Returns:
        - array: Predicted values.
        """
        predictions = np.zeros(X.shape[0])
        for estimator in self.estimators_:
            predictions += estimator.predict(X)
        return predictions / self.n_estimators

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters:
        - X (array-like): Test samples.
        - y (array-like): True labels for X.

        Returns:
        - float: R^2 of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'criterion': self.criterion,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'min_weight_fraction_leaf': self.min_weight_fraction_leaf,
            'max_features': self.max_features,
            'max_leaf_nodes': self.max_leaf_nodes,
            'min_impurity_decrease': self.min_impurity_decrease,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'n_jobs': self.n_jobs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'ccp_alpha': self.ccp_alpha,
            'max_samples': self.max_samples,
            'monotonic_cst': self.monotonic_cst,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

