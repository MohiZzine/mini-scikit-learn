import numpy as np
from scipy import sparse

import os
import sys

# Add the base module to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base.base_scaler import BaseScaler

class MaxAbsScaler(BaseScaler):
    """
    Scale each feature by its maximum absolute value.

    This scaler scales and translates each feature individually such that the maximal absolute value
    of each feature in the training set will be 1.0. It does not shift/center the data, and thus does not destroy any sparsity.

    Parameters
    ----------
    copy : bool, default=True
        Set to False to perform inplace scaling and avoid a copy (if the input is already a numpy array).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data.

    max_abs_ : ndarray of shape (n_features,)
        Per feature maximum absolute value.
    """

    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        """
        Compute the maximum absolute value to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum used for later scaling along the features axis.
        y : None
            Ignored.
        """
        X = self._validate_data(X)
        self.max_abs_ = np.max(np.abs(X), axis=0)
        self.scale_ = self.max_abs_
        return self

    def transform(self, X):
        """
        Scale the data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data that should be scaled.
        """
        X = self._validate_data(X, copy=self.copy)
        X /= self.scale_
        return X

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params and returns a transformed version of X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            Target values (None for unsupervised transformations).
        """
        return self.fit(X).transform(X)

    def _validate_data(self, X, copy=True):
        """
        Validate input data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to validate.
        copy : bool, default=True
            Set to False to perform inplace scaling and avoid a copy (if the input is already a numpy array).
        """
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = np.array(X, copy=copy, dtype=np.float64)
        return X
