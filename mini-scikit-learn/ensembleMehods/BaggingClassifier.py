import numpy as np
import os
import sys

from sklearn.base import clone
from sklearn.utils import check_random_state
from SupervisedLearning.DecisionTrees.DecisionTreeClassifier import DecisionTreeClassifier

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class BaggingClassifier:
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 oob_score=False, warm_start=False,
                 n_jobs=None, random_state=None, verbose=0):
        self.estimator = estimator if estimator is not None else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.estimators_ = []
        self.estimators_samples_ = []
        self.estimators_features_ = []
        self.oob_decision_function_ = None
        self.classes_ = None

    def _generate_sample_indices(self, random_state, n_samples):
        """Generate indices for dataset sampling."""
        if self.bootstrap:
            return random_state.randint(0, n_samples, n_samples)
        else:
            return random_state.permutation(n_samples)[:n_samples]

    def _generate_feature_indices(self, random_state, n_features):
        """Generate indices for feature sampling."""
        if self.bootstrap_features:
            return random_state.randint(0, n_features, n_features)
        else:
            return random_state.permutation(n_features)[:n_features]

    def fit(self, X, y):
        """Build a Bagging ensemble of estimators
        from the training set (X, y)."""
        n_samples, n_features = X.shape
        self.estimators_ = []
        self.estimators_samples_ = []
        self.estimators_features_ = []
        self.classes_ = np.unique(y)

        random_state = check_random_state(self.random_state)
        for i in range(self.n_estimators):
            estimator = clone(self.estimator)
            sample_indices = self._generate_sample_indices(random_state,
                                                           n_samples)
            feature_indices = self._generate_feature_indices(random_state,
                                                             n_features)
            self.estimators_samples_.append(sample_indices)
            self.estimators_features_.append(feature_indices)
            X_sample = X[sample_indices][:, feature_indices]
            y_sample = y[sample_indices]
            estimator.fit(X_sample, y_sample)
            self.estimators_.append(estimator)

        if self.oob_score:
            self._set_oob_score(X, y)
        return self

    def _set_oob_score(self, X, y):
        """Calculate out-of-bag score."""
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.classes_)))
        for estimator, samples in zip(self.estimators_, self.estimators_samples_):
            mask = np.ones(n_samples, dtype=bool)
            mask[samples] = False
            if np.any(mask):
                oob_predictions = estimator.predict(X[mask])
                for i, pred in zip(np.where(mask)[0], oob_predictions):
                    predictions[i, self.classes_ == pred] += 1
        self.oob_decision_function_ = predictions / \
            np.sum(predictions, axis=1, keepdims=True)
        self.oob_score_ = np.mean(np.argmax(predictions, axis=1) == y)

    def predict(self, X):
        """Predict class for X."""
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, len(self.classes_)))
        for estimator in self.estimators_:
            preds = estimator.predict(X)
            for i, pred in enumerate(preds):
                predictions[i, self.classes_ == pred] += 1
        return self.classes_[np.argmax(predictions, axis=1)]

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, len(self.classes_)))
        for estimator in self.estimators_:
            probas += estimator.predict_proba(X)
        return probas / len(self.estimators_)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels."""
        return np.mean(self.predict(X) == y)
