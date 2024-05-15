import numpy as np

class StandardScaler:
    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.copy = copy
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        if self.copy:
            X = X.astype(float)
        X = np.array(X, copy=self.copy)
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        if self.with_std:
            self.var_ = np.var(X, axis=0)
            self.scale_ = np.sqrt(self.var_)
        
        return self

    def transform(self, X):
        if self.copy:
          X = X.astype(float)
        X = np.array(X, copy=self.copy)

        if self.copy:
            X = X.astype(float)
        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1), copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        self.scale_ = None
        self.min_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0

    def fit(self, X, y=None):
        if self.copy:
            X = X.astype(float)
        X = np.array(X, copy=self.copy)
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = X.shape[0]
        
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        range_min, range_max = self.feature_range
        
        self.scale_ = (range_max - range_min) / (self.data_range_ + np.finfo(float).eps)
        self.min_ = range_min - self.data_min_ * self.scale_
        
        return self

    def transform(self, X):
        if self.copy:
            X = X.astype(float)
        X = np.array(X, copy=self.copy)
        X *= self.scale_
        X += self.min_
        
        if self.clip:
            X = np.clip(X, self.feature_range[0], self.feature_range[1])
        
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)



class RobustScaler:
    def __init__(self, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)):
        """
        Initializes the RobustScaler with options for centering and scaling
        using the interquartile range.

        Parameters:
        - with_centering: bool, default True
            Whether to center the data before scaling.
        - with_scaling: bool, default True
            Whether to scale data to interquartile range.
        - quantile_range: tuple, default (25.0, 75.0)
            Quantile range used for scaling. Uses the lower and upper quartile values.
        """
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        """
        Fit the RobustScaler to the data.

        Parameters:
        - X : array-like, shape [n_samples, n_features]
            The data used to compute the median and the quantile range used
            for scaling along each feature.

        Returns:
        - self : object
            Returns the instance itself.
        """
        q_min, q_max = self.quantile_range
        self.center_ = np.median(X, axis=0) if self.with_centering else None
        if self.with_scaling:
            q1 = np.percentile(X, q_min, axis=0)
            q3 = np.percentile(X, q_max, axis=0)
            self.scale_ = q3 - q1
        return self

    def transform(self, X):
        """
        Transform the data using the fitted median and interquartile range.

        Parameters:
        - X : array-like, shape [n_samples, n_features]
            The data that should be transformed.

        Returns:
        - X_tr : array-like, shape [n_samples, n_features]
            The transformed data.
        """
        if self.center_ is not None:
            X = X - self.center_
        if self.scale_ is not None:
            X = X / self.scale_
        return X
