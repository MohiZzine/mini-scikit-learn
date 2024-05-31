# File: preprocessing/ordinal_encoder.py

import numpy as np

class OrdinalEncoder:
    """
    Encode categorical features as an integer array.

    Parameters
    ----------
    categories : 'auto' or list of array-like, default='auto'
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : `categories[i]` holds the categories expected in the `i`th column.

    dtype : number type, default=np.float64
        Desired dtype of output.

    handle_unknown : {'error', 'use_encoded_value'}, default='error'
        When set to 'error' an error will be raised in case an unknown categorical feature is present during transform. 
        When set to 'use_encoded_value', the encoded value of unknown categories will be set to the value given for the parameter `unknown_value`.

    unknown_value : int or np.nan, default=None
        When the parameter `handle_unknown` is set to 'use_encoded_value', this parameter is required and will set the encoded value of unknown categories.
        It has to be distinct from the values used to encode any of the categories in `fit`.

    encoded_missing_value : int or np.nan, default=np.nan
        Encoded value of missing categories. If set to np.nan, then the dtype parameter must be a float dtype.

    min_frequency : int or float, default=None
        Specifies the minimum frequency below which a category will be considered infrequent.

    max_categories : int, default=None
        Specifies an upper limit to the number of output categories for each input feature when considering infrequent categories.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fit (in order of the features in X and corresponding with the output of transform).

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    infrequent_categories_ : list of ndarray
        Infrequent categories for each feature.
    """

    def __init__(self, categories='auto', dtype=np.float64, handle_unknown='error', unknown_value=None, encoded_missing_value=np.nan, min_frequency=None, max_categories=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.min_frequency = min_frequency
        self.max_categories = max_categories
        self.categories_ = None
        self.n_features_in_ = None
        self.feature_names_in_ = None
        self.infrequent_categories_ = None

    def fit(self, X, y=None):
        """
        Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to determine the categories of each feature.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted encoder.
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        if self.categories == 'auto':
            self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        else:
            self.categories_ = self.categories

        return self

    def transform(self, X):
        """
        Transform X to ordinal codes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to encode.

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features)
            Transformed input.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        X_out = np.empty_like(X, dtype=self.dtype)
        for i in range(n_features):
            cat_mapping = {cat: idx for idx, cat in enumerate(self.categories_[i])}
            for j in range(n_samples):
                value = X[j, i]
                if value in cat_mapping:
                    X_out[j, i] = cat_mapping[value]
                elif self.handle_unknown == 'use_encoded_value':
                    X_out[j, i] = self.unknown_value
                else:
                    raise ValueError(f"Unknown category {value} in column {i}")

        return X_out

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X):
        """
        Convert the data back to the original representation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_encoded_features)
            The transformed data.

        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Inverse transformed array.
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        X_tr = np.empty_like(X, dtype=object)
        for i in range(n_features):
            categories = self.categories_[i]
            for j in range(n_samples):
                idx = X[j, i]
                if np.isnan(idx):
                    X_tr[j, i] = None
                else:
                    X_tr[j, i] = categories[int(idx)]

        return X_tr

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
            "categories": self.categories,
            "dtype": self.dtype,
            "handle_unknown": self.handle_unknown,
            "unknown_value": self.unknown_value,
            "encoded_missing_value": self.encoded_missing_value,
            "min_frequency": self.min_frequency,
            "max_categories": self.max_categories
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
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
