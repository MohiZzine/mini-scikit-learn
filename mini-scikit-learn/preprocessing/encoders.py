import numpy as np
from scipy import sparse


class LabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        """Fit label encoder to y."""
        unique_classes = np.unique(y)
        self.classes_ = unique_classes
        return self

    def transform(self, y):
        """Transform labels to normalized encoding."""
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance is not fitted yet.")
        
        # Create a dictionary mapping class labels to their integer encoding
        class_lookup = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        # Map the input labels to their respective integer labels
        encoded_labels = np.array([class_lookup[label] for label in y], dtype=int)
        return encoded_labels

    def inverse_transform(self, y_enc):
        """Transform labels back to original encoding."""
        if self.classes_ is None:
            raise ValueError("This LabelEncoder instance is not fitted yet.")
        
        # Create a dictionary mapping integer encoding back to original class labels
        inverse_lookup = {idx: cls for idx, cls in enumerate(self.classes_)}
        
        # Map the encoded labels back to original labels
        original_labels = np.array([inverse_lookup[label] for label in y_enc])
        return original_labels

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels."""
        return self.fit(y).transform(y)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {"deep": deep}  # Simplified to illustrate

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class OneHotEncoder:
    def __init__(self, categories='auto', drop=None, sparse_output=True, dtype=np.float64):
        self.categories = categories
        self.drop = drop
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.category_indices_ = None

    def fit(self, X, y=None):
        """Fit OneHotEncoder to X by detecting categories."""
        if self.categories == 'auto':
            self.categories_ = [np.unique(col) for col in X.T]
        else:
            self.categories_ = self.categories
        self.category_indices_ = {category: idx for idx, category in enumerate(self.categories_)}
        return self

    def transform(self, X):
        """Transform X using one-hot encoding."""
        if not self.categories_:
            raise ValueError("This OneHotEncoder instance is not fitted yet.")

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
            indices = X[:, i*len(categories):(i+1)*len(categories)].argmax(axis=1)
            X_transformed[:, i] = [categories[idx] for idx in indices]

        return X_transformed


class OrdinalEncoder:
    def __init__(self, categories='auto', dtype=np.float64, handle_unknown='error', unknown_value=None):
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.category_indices_ = None

    def fit(self, X, y=None):
        """Fit the OrdinalEncoder to X."""
        if self.categories == 'auto':
            self.categories_ = [np.unique(col) for col in X.T]
        else:
            self.categories_ = self.categories
        return self

    def transform(self, X):
        """Transform X to ordinal codes."""
        if self.categories_ is None:
            raise ValueError("This OrdinalEncoder instance is not fitted yet.")

        X_encoded = np.zeros(X.shape, dtype=self.dtype)
        for i, categories in enumerate(self.categories_):
            category_map = {category: idx for idx, category in enumerate(categories)}
            for j, item in enumerate(X[:, i]):
                if item in category_map:
                    X_encoded[j, i] = category_map[item]
                elif self.handle_unknown == 'error':
                    raise ValueError(f"Unknown category '{item}' found during transform.")
                else:
                    X_encoded[j, i] = self.unknown_value

        return X_encoded

    def inverse_transform(self, X):
        """Convert the data back to the original representation."""
        X_inv = np.empty_like(X, dtype=object)
        for i, categories in enumerate(self.categories_):
            for j, code in enumerate(X[:, i]):
                if code < len(categories):
                    X_inv[j, i] = categories[int(code)]
                else:
                    X_inv[j, i] = None
        return X_inv

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'categories': self.categories,
            'dtype': self.dtype,
            'handle_unknown': self.handle_unknown,
            'unknown_value': self.unknown_value
        }

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
