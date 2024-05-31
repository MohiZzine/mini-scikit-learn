import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

class IterativeImputer:
    """
    IterativeImputer for multivariate imputation.

    Parameters:
    - estimator: Estimator for predicting missing values. Default is LinearRegression.
    - missing_values: Placeholder for missing values. Default is np.nan.
    - sample_posterior: Sample from Gaussian posterior for each imputation. Default is False.
    - max_iter: Maximum number of imputation iterations. Default is 10.
    - tol: Tolerance to declare convergence. Default is 0.001.
    - n_nearest_features: Number of nearest features to use for imputation. Default is None.
    - initial_strategy: Initial imputation strategy ("mean", "median", "most_frequent"). Default is 'mean'.
    - imputation_order: Order of imputation ("ascending", "descending", "roman", "arabic", "random"). Default is 'ascending'.
    - verbose: Verbosity level. Default is 0.
    - random_state: Random state for reproducibility. Default is None.
    - min_value: Minimum possible imputed value. Default is -np.inf.
    - max_value: Maximum possible imputed value. Default is np.inf.
    """

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
        self.random_state_ = check_random_state(random_state)

    def fit(self, X, y=None):
        """
        Fit the IterativeImputer to the data.

        Parameters:
        - X : array-like of shape (n_samples, n_features)
        - y : Ignored

        Returns:
        - self : object
        """
        X = np.asarray(X, dtype=np.float64)
        if self.initial_strategy == "mean":
            self.initial_statistics_ = np.nanmean(X, axis=0)
        elif self.initial_strategy == "median":
            self.initial_statistics_ = np.nanmedian(X, axis=0)
        elif self.initial_strategy == "most_frequent":
            self.initial_statistics_ = stats.mode(X, nan_policy='omit')[0][0]
        else:
            raise ValueError(f"Unknown strategy {self.initial_strategy}")
        
        self.n_features_ = X.shape[1]
        self.imputation_sequence_ = self._get_feature_order(self.n_features_)
        
        self.statistics_ = np.copy(self.initial_statistics_)
        X_imputed = self.transform(X)
        self.statistics_ = None  # Clear after fit

        return self

    def _get_feature_order(self, n_features):
        order = np.arange(n_features)
        if self.imputation_order == 'ascending':
            order = order[np.argsort(self.initial_statistics_)]
        elif self.imputation_order == 'descending':
            order = order[np.argsort(-self.initial_statistics_)]
        elif self.imputation_order == 'roman':
            order = order
        elif self.imputation_order == 'arabic':
            order = order[::-1]
        elif self.imputation_order == 'random':
            self.random_state_.shuffle(order)
        else:
            raise ValueError("Invalid imputation order")
        return order

    def transform(self, X):
        """
        Apply iterative imputation to the provided data matrix.

        Parameters:
        - X : array-like of shape (n_samples, n_features)

        Returns:
        - X_transformed : array-like of shape (n_samples, n_features)
        """
        X = np.asarray(X, dtype=np.float64)
        mask = np.isnan(X)
        X_filled = np.copy(X)

        for _ in range(self.max_iter):
            X_prev = X_filled.copy()
            for i in self.imputation_sequence_:
                y_train = X_filled[~mask[:, i], i]
                X_train = X_filled[~mask[:, i]][:, self.imputation_sequence_]
                X_test = X_filled[mask[:, i]][:, self.imputation_sequence_]

                if X_train.shape[0] > 0 and X_test.shape[0] > 0:
                    self.estimator.fit(X_train, y_train)
                    X_filled[mask[:, i], i] = self.estimator.predict(X_test)
                    X_filled[:, i] = np.clip(X_filled[:, i], self.min_value, self.max_value)

            if np.all(np.abs(X_filled - X_prev) < self.tol):
                break

        return X_filled

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
