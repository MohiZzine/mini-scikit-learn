import numpy as np
from itertools import product
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from metrics.classification.accuracy import accuracy_score
from metrics.classification.precision_score import precision_score
from metrics.classification.f1_score import f1_score
from metrics.classification.recall_score import recall_score

class GridSearchCV:
    """
    Grid search on hyperparameters.

    Exhaustive search over specified parameter values for an estimator.
    
    Parameters
    ----------
    estimator : estimator object
        This is assumed to implement the scikit-learn estimator interface.
    param_grid : dict
        Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    scoring : str, default='accuracy'
        A string specifying the scoring metric.
    cv : cross-validation generator, default=None
        Determines the cross-validation splitting strategy.
    """
    def __init__(self, estimator, param_grid, scoring='accuracy', cv=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, X, y):
        """
        Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
        """
        param_grid = list(ParameterGrid(self.param_grid))
        results = []
        best_score = -np.inf
        
        for params in param_grid:
            self.estimator.set_params(**params)
            scores = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(scores)
            results.append((params, mean_score))

            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = best_score
                self.best_estimator_ = clone(self.estimator).set_params(**params)
        
        self.cv_results_ = results
        self.best_estimator_.fit(X, y)

    def predict(self, X):
        """
        Call predict on the estimator with the best found parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        """
        Returns the score on the given data, using the scoring method specified.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True values for X.
        
        Returns
        -------
        score : float
            Score of the estimator.
        """
        return self.best_estimator_.score(X, y)

class ParameterGrid:
    """
    Grid of parameters with a discrete number of values for each.

    Parameters
    ----------
    param_grid : dict
        Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    """
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        """
        Iterate over parameter settings.
        
        Yields
        ------
        params : dict
            Dictionary of parameter settings.
        """
        keys, values = zip(*self.param_grid.items())
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params

def cross_val_score(model, X, y, cv, scoring='accuracy'):
    """
    Evaluate a score by cross-validation.

    Parameters
    ----------
    model : estimator object
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,)
        The target variable to try to predict in the case of supervised learning.
    cv : cross-validation generator
        The cross-validation splitting strategy.
    scoring : str, default='accuracy'
        A string specifying the scoring metric.
    
    Returns
    -------
    scores : ndarray of shape (n_splits,)
        Array of scores of the estimator for each run of the cross-validation.
    """
    if scoring == 'accuracy':
        scoring_func = accuracy_score
    elif scoring == 'precision':
        scoring_func = precision_score
    elif scoring == 'recall':
        scoring_func = recall_score
    elif scoring == 'f1':
        scoring_func = f1_score
    else:
        raise ValueError("Unsupported scoring method.")

    scores = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scoring_func(y_test, y_pred)
        scores.append(score)
    return np.array(scores)

def clone(estimator):
    """
    Clone a given estimator.

    Parameters
    ----------
    estimator : estimator object
        The estimator to be cloned.
    
    Returns
    -------
    estimator_clone : estimator object
        The cloned estimator.
    """
    estimator_clone = estimator.__class__()
    estimator_clone.__dict__.update(estimator.__dict__)
    return estimator_clone
