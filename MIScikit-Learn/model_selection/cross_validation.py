import numpy as np

class KFold:
    """
    K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds.
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.
    random_state : int or None, default=None
        When shuffle is True, random_state affects the ordering of the indices.
    """
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        Yields
        ------
        train_indices : ndarray
            The training set indices for that split.
        test_indices : ndarray
            The testing set indices for that split.
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        folds = []
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            folds.append(indices[start:stop])
            current = stop
        
        for i in range(self.n_splits):
            train_indices = np.hstack([folds[j] for j in range(self.n_splits) if j != i])
            test_indices = folds[i]
            yield train_indices, test_indices

def cross_val_score(model, X, y, cv):
    """
    Evaluate a score by cross-validation.

    Parameters
    ----------
    model : estimator object
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,)
        The target variable to try to predict in the case of supervised learning.
    cv : cross-validation generator
        The cross-validation splitting strategy.
    
    Returns
    -------
    scores : ndarray of shape (n_splits,)
        Array of scores of the estimator for each run of the cross-validation.
    """
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    return np.array(scores)
