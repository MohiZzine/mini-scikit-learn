import numpy as np
from collections import Counter
from itertools import chain

class StratifiedKFold:
    """
    Stratified K-Folds cross-validator

    Provides train/test indices to split data into train/test sets.
    This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
        
    shuffle : boolean, default=False
        Whether to shuffle each class's samples before splitting into batches.
        
    random_state : int or None, default=None
        When shuffle is True, random_state affects the ordering of the indices, which controls the randomness of each fold.
        Otherwise, this parameter has no effect.
    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is the number of features.
        
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        
        test : ndarray
            The testing set indices for that split.
        """
        y = np.asarray(y)
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        # Determine the unique classes and their counts
        unique_classes, y_indices = np.unique(y, return_inverse=True)
        class_counts = np.bincount(y_indices)
        n_classes = len(unique_classes)
        
        # Determine the class indices
        class_indices = [np.where(y_indices == i)[0] for i in range(n_classes)]

        # Initialize the folds
        folds = [[] for _ in range(self.n_splits)]
        
        # Distribute the samples to the folds
        for class_idx in range(n_classes):
            fold_indices = np.array_split(class_indices[class_idx], self.n_splits)
            for fold_idx in range(self.n_splits):
                folds[fold_idx].extend(fold_indices[fold_idx])

        # Convert fold lists to numpy arrays
        folds = [np.array(fold) for fold in folds]

        for fold in range(self.n_splits):
            test_idx = folds[fold]
            train_idx = np.hstack([folds[i] for i in range(self.n_splits) if i != fold])
            yield train_idx, test_idx

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold + 1}")
        print(f"Train indices: {train_idx}")
        print(f"Test indices: {test_idx}")
        print(f"Train class distribution: {Counter(y[train_idx])}")
        print(f"Test class distribution: {Counter(y[test_idx])}")
        print()