import numpy as np
from scipy import sparse


class OneHotEncoder:
    def __init__(self, categories='auto', drop=None, sparse_output=True, dtype=np.float64):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.categories_ = None

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X by detecting categories."""
        X = np.asarray(X)
        if self.categories == 'auto':
            self.categories_ = [np.unique(col) for col in X.T]
        else:
            self.categories_ = self.categories
        return self

    def transform(self, X):
        """Transform X using one-hot encoding."""
        if self.categories_ is None:
            raise ValueError("This OneHotEncoder instance is not fitted yet.")

        X = np.asarray(X)
        # List to hold column-wise one-hot encodings
        encoded_features = []

        for j, categories in enumerate(self.categories_):
            column = X[:, j]
            feature = np.zeros((column.shape[0], len(categories)), dtype=self.dtype)

            for i, category in enumerate(categories):
                feature[:, i] = (column == category).astype(self.dtype)

            encoded_features.append(feature)

        # Concatenate all the feature columns
        encoded_array = np.hstack(encoded_features)

        if self.sparse_output:
            return sparse.csr_matrix(encoded_array)
        else:
            return encoded_array

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        """Convert the data back to the original representation."""
        if isinstance(X, sparse.csr_matrix):
            X = X.toarray()

        X_transformed = np.zeros((X.shape[0], len(self.categories_)), dtype=object)

        for i, categories in enumerate(self.categories_):
            indices = X[:, i * len(categories):(i + 1) * len(categories)].argmax(axis=1)
            X_transformed[:, i] = [categories[idx] for idx in indices]

        return X_transformed