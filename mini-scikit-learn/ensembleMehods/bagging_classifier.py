import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample

class BaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    A Bagging classifier.

    Parameters
    ----------
    base_estimator : object
        The base estimator to fit on random subsets of the dataset.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : float, default=1.0
        The proportion of the dataset to include in each random subset.

    random_state : int, default=None
        Controls the random resampling of the original dataset.
    """

    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        self.estimators_ = []
        np.random.seed(self.random_state)
        
        n_samples = int(self.max_samples * X.shape[0])
        
        for _ in range(self.n_estimators):
            estimator = clone(self.base_estimator)
            X_resampled, y_resampled = resample(X, y, n_samples=n_samples, random_state=self.random_state)
            estimator.fit(X_resampled, y_resampled)
            self.estimators_.append(estimator)
        
        return self

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def predict_proba(self, X):
        probas = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        return np.mean(probas, axis=0)
