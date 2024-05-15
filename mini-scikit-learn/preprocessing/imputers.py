import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.utils import check_random_state, check_array, Bunch
from sklearn.metrics import pairwise_distances
from scipy import stats

class SimpleImputer:
    def __init__(self, missing_values=np.nan, strategy="mean", fill_value=None):
        """
        Initialize the SimpleImputer with the specified imputation strategy.

        Parameters:
        - missing_values: The placeholder for the missing values. All occurrences of
                          missing_values will be imputed. For pandas' dataframes, this should
                          be np.nan, while for numpy arrays, it can be None.
        - strategy: The imputation strategy:
            - "mean": Replace missing using the mean along each column. Can only be used with numeric data.
            - "median": Replace missing using the median along each column. Can only be used with numeric data.
            - "most_frequent": Replace missing using the most frequent value along each column.
            - "constant": Replace missing with fill_value. Can be used with strings or numeric data.
        - fill_value: When strategy == "constant", fill_value is used to replace all occurrences of missing_values.
                      If left to None, fill_value will be 0 when imputing numerical data and "missing_value" for strings.
        """
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X, y=None):
        """
        Fit the imputer on the data by calculating the statistics (mean, median, or most frequent) used to impute missing values.

        - X : array-like of shape (n_samples, n_features)
        """
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            self.statistics_ = stats.mode(X, nan_policy='omit').mode[0]
        elif self.strategy != "constant":
            raise ValueError(f"Unknown strategy {self.strategy}")

        return self

    def transform(self, X):
        """
        Apply the imputation strategy to the provided data matrix.

        - X : array-like of shape (n_samples, n_features)
        """
        if X is None:
            raise ValueError("Data passed to transform() cannot be None.")

        X_transformed = np.array(X, copy=True)
        if self.strategy == "constant" and self.fill_value is not None:
            fill_values = np.full(X_transformed.shape[1], self.fill_value)
            mask = np.isnan(X_transformed)
            X_transformed[mask] = fill_values[mask]
        else:
            for i in range(X_transformed.shape[1]):
                mask = np.isnan(X_transformed[:, i])
                X_transformed[mask, i] = self.statistics_[i]

        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class IterativeImputer:
    def __init__(self, estimator=None, missing_values=np.nan, sample_posterior=False,
                 max_iter=10, tol=0.001, n_nearest_features=None, initial_strategy='mean',
                 imputation_order='ascending', verbose=0, random_state=None,
                 min_value=-np.inf, max_value=np.inf):
        self.estimator = estimator if estimator is not None else LinearRegression()
        self.missing_values = missing_values
        self.sample_posterior = sample_posterior
        self.max_iter = max_iter
        self.tol = tol
        self.n_nearest_features = n_nearest_features
        self.initial_strategy = initial_strategy
        self.imputation_order = imputation_order
        self.verbose = verbose
        self.random_state = random_state
        self.min_value = min_value
        self.max_value = max_value
        self.initial_imputer = SimpleImputer(strategy=self.initial_strategy, missing_values=missing_values)
        self.random_state_ = check_random_state(random_state)

    def _get_feature_order(self, n_features):
        if self.imputation_order in ['ascending', 'descending', 'roman', 'arabic', 'random']:
            if self.imputation_order == 'ascending':
                return np.argsort(np.sum(np.isnan(self.initial_imputer.statistics_), axis=0))
            elif self.imputation_order == 'descending':
                return np.argsort(-np.sum(np.isnan(self.initial_imputer.statistics_), axis=0))
            elif self.imputation_order == 'roman':
                return np.arange(n_features)
            elif self.imputation_order == 'arabic':
                return np.arange(n_features)[::-1]
            elif self.imputation_order == 'random':
                order = np.arange(n_features)
                self.random_state_.shuffle(order)
                return order
        else:
            raise ValueError("Invalid imputation order")

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.initial_imputer.fit(X)
        X_imputed = self.initial_imputer.transform(X)
        self.feature_order_ = self._get_feature_order(X.shape[1])

        self.n_iter_ = 0
        for i in range(self.max_iter):
            X_previous = X_imputed.copy()
            X_imputed = self._impute_round(X_imputed)
            if self.verbose > 0:
                print(f"Iteration {i}: max change = {np.max(np.abs(X_imputed - X_previous))}")
            if np.all(np.abs(X_imputed - X_previous) < self.tol):
                break
            self.n_iter_ += 1

        return self

    def _impute_round(self, X):
        for feature_idx in self.feature_order_:
            missing_idx = np.isnan(X[:, feature_idx])
            if np.any(missing_idx):
                observed_idx = ~missing_idx
                X_train = X[observed_idx][:, self.feature_order_]
                y_train = X[observed_idx, feature_idx]
                X_test = X[missing_idx][:, self.feature_order_]
                self.estimator.fit(X_train, y_train)
                X[missing_idx, feature_idx] = self.estimator.predict(X_test)
                # Apply min/max constraints
                X[:, feature_idx] = np.clip(X[:, feature_idx], self.min_value, self.max_value)
        return X

    def transform(self, X):
        X_imputed = np.array(X, copy=True)
        for i in range(self.max_iter):
            X_imputed = self._impute_round(X_imputed)
        return X_imputed

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class KNNImputer:
    def __init__(self, n_neighbors=5, weights='uniform', metric='nan_euclidean', missing_values=np.nan, copy=True, add_indicator=False):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.missing_values = missing_values
        self.copy = copy
        self.add_indicator = add_indicator
        self.output_format = 'default'  # 'pandas' or 'polars' could be other options

    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=False, dtype=np.float64, force_all_finite=False, copy=self.copy)
        self.X_fit_ = X
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = ['x{}'.format(i) for i in range(X.shape[1])]
        return self

    def transform(self, X):
        X = check_array(X, accept_sparse=False, dtype=np.float64, force_all_finite=False, copy=self.copy)
        return self._impute(X)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X).transform(X)

    def _impute(self, X):
        for i in range(X.shape[1]):
            missing = np.isnan(X[:, i])
            if np.any(missing):
                distances = pairwise_distances(X[missing, :], self.X_fit_, metric=self.metric)
                ind = np.argsort(distances, axis=1)[:, :self.n_neighbors]
                weights = np.ones_like(ind) if self.weights == 'uniform' else 1 / (distances[:, ind] + 1e-5)
                values = self.X_fit_[ind, i]
                X[missing, i] = np.sum(weights * values, axis=1) / np.sum(weights, axis=1)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.feature_names_in_)
        elif isinstance(input_features, (list, np.ndarray)):
            if len(input_features) != len(self.feature_names_in_):
                raise ValueError("input_features length must match number of features during fit.")
            return np.array(input_features)
        else:
            raise TypeError("input_features must be list or numpy ndarray.")

    def get_params(self, deep=True):
        return {'n_neighbors': self.n_neighbors, 'weights': self.weights, 'metric': self.metric, 'missing_values': self.missing_values, 'copy': self.copy, 'add_indicator': self.add_indicator}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def set_output(self, transform=None):
        valid_outputs = ['default', 'pandas', 'polars']
        if transform in valid_outputs:
            self.output_format = transform
        else:
            raise ValueError("transform must be one of: {}".format(', '.join(valid_outputs)))
        return self