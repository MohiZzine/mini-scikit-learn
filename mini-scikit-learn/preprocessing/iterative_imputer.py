import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.utils import check_array

class IterativeImputer:
    """
    Iterative imputer for completing missing values using a specified estimator.

    Parameters
    ----------
    estimator : object, default=BayesianRidge()
        The estimator to use for the iterative imputation.

    max_iter : int, default=10
        The maximum number of imputation iterations.

    tol : float, default=1e-3
        Tolerance to declare convergence for the stopping condition.

    random_state : int, RandomState instance or None, default=None
        Random number seed for reproducibility.
    """

    def __init__(self, estimator=None, max_iter=10, tol=1e-3, random_state=None):
        self.estimator = estimator if estimator is not None else BayesianRidge()
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.statistics_ = None

    def fit(self, X, y=None):
        """
        Fit the imputer on X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        y : Ignored

        Returns
        -------
        self : object
            Fitted imputer.
        """
        X = check_array(X, dtype=np.float64, force_all_finite=False)
        self.statistics_ = np.nanmean(X, axis=0)

        for _ in range(self.max_iter):
            X_old = X.copy()
            for i in range(X.shape[1]):
                mask = np.isnan(X[:, i])
                if np.any(mask):
                    X[mask, i] = self._impute_column(X[:, i], X[:, ~mask])
            if np.all(np.abs(X - X_old) < self.tol):
                break

        return self

    def _impute_column(self, y, X):
        mask = ~np.isnan(y)
        self.estimator.fit(X[mask], y[mask])
        return self.estimator.predict(X[~mask])

    def transform(self, X):
        """
        Impute all missing values in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        X_imputed : array of shape (n_samples, n_features)
            The imputed dataset.
        """
        X = check_array(X, dtype=np.float64, force_all_finite=False)
        X_imputed = X.copy()

        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            if np.any(mask):
                X_imputed[mask, i] = self._impute_column(X[:, i], X[:, ~mask])

        return X_imputed

    def fit_transform(self, X, y=None):
        """
        Fit the imputer on X, then transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to complete.

        y : Ignored

        Returns
        -------
        X_imputed : array of shape (n_samples, n_features)
            The imputed dataset.
        """
        return self.fit(X, y).transform(X)

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            "estimator": self.estimator,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
