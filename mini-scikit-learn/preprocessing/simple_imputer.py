import numpy as np
from scipy import stats

class SimpleImputer:
    """
    SimpleImputer for completing missing values using simple strategies.

    Parameters:
    - missing_values: Placeholder for missing values. Default is np.nan.
    - strategy: The imputation strategy ("mean", "median", "most_frequent", "constant").
    - fill_value: Value to use when strategy="constant". Default is None.
    - copy: If True, a copy of X will be created. Default is True.
    """

    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None, copy=True):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.copy = copy
        self.statistics_ = None

    def fit(self, X, y=None):
        """
        Fit the imputer on the data by calculating the statistics used to impute missing values.

        Parameters:
        - X : array-like of shape (n_samples, n_features)

        Returns:
        - self : object
        """
        X = np.array(X, dtype=np.float64)
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            self.statistics_ = stats.mode(X, nan_policy='omit').mode[0]
        elif self.strategy == "constant":
            if self.fill_value is None:
                self.fill_value = 0
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")
        return self

    def transform(self, X):
        """
        Apply the imputation strategy to the provided data matrix.

        Parameters:
        - X : array-like of shape (n_samples, n_features)

        Returns:
        - X_transformed : array-like of shape (n_samples, n_features)
        """
        if self.statistics_ is None:
            raise ValueError("SimpleImputer is not fitted yet.")
        
        X = np.array(X, dtype=np.float64, copy=self.copy)
        mask = np.isnan(X)
        for i in range(X.shape[1]):
            X[mask[:, i], i] = self.statistics_[i]
        
        return X

    def fit_transform(self, X, y=None):
        """
        Fit to data, then transform it.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
        - y : Ignored

        Returns:
        - X_new : array-like of shape (n_samples, n_features)
        """
        return self.fit(X).transform(X)
