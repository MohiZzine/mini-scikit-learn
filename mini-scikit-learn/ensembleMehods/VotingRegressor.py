import numpy as np
from sklearn.base import clone

class VotingRegressor:
    def __init__(self, estimators, weights=None, n_jobs=None, verbose=False):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        self.estimators_ = []
        self.named_estimators_ = {}
        
        for name, estimator in self.estimators:
            if estimator == 'drop':
                continue
            cloned_estimator = clone(estimator)
            cloned_estimator.fit(X, y, sample_weight=sample_weight)
            self.estimators_.append(cloned_estimator)
            self.named_estimators_[name] = cloned_estimator
        
        return self

    def predict(self, X):
        predictions = np.asarray([estimator.predict(X) for estimator in self.estimators_]).T
        if self.weights is not None:
            avg = np.average(predictions, axis=1, weights=self.weights)
        else:
            avg = np.mean(predictions, axis=1)
        return avg

    def get_params(self, deep=True):
        return {
            'estimators': self.estimators,
            'weights': self.weights,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
