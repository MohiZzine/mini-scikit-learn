import numpy as np

class Normalizer:
    def __init__(self, norm='l2', copy=True):
        """
        Initialize the Normalizer class with options for the norm, copying behavior, and axis.

        Parameters:
        - norm : str, {'l1', 'l2', 'max'}, default='l2'
            The norm to use to normalize each non-zero sample or feature.
        - copy : bool, default=True
            Set to False to perform inplace row normalization and avoid a copy (if the input
            is already a numpy array and if axis is 1).
        """
        self.norm = norm
        self.copy = copy

    def fit(self, X, y=None):
        # No fitting process needed for normalization
        return self

    def transform(self, X, axis=1, return_norm=False):
        """
        Scale input vectors individually to unit norm.

        Parameters:
        - X : array-like, shape [n_samples, n_features]
            The data to normalize, element by element.
        - axis : {0, 1}, default=1
            The axis used to normalize the data along. If 1, independently normalize each sample,
            otherwise (if 0) normalize each feature.
        - return_norm : bool, default=False
            Whether to return the computed norms along with the normalized array.

        Returns:
        - X_normalized : array, shape [n_samples, n_features]
            Normalized input X.
        - norms : array, [n_samples] or [n_features]
            The norms of the vectors (only returned if return_norm is True).
        """
        X = np.array(X, dtype=float, copy=self.copy)  # Optionally copy the data
        
        if self.norm == 'l2':
            norms = np.linalg.norm(X, ord=2, axis=axis, keepdims=True)
        elif self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=axis, keepdims=True)
        elif self.norm == 'max':
            norms = np.max(np.abs(X), axis=axis, keepdims=True)
        else:
            raise ValueError("norm should be 'l1', 'l2', or 'max'.")

        X_normalized = X / norms
        if return_norm:
            return X_normalized, norms.squeeze()
        else:
            return X_normalized
