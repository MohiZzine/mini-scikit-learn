import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score as sklearn_cross_val_score, KFold as SklearnKFold

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
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
    scores = []
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    return np.array(scores)

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Initialize the custom k-fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform custom cross-validation
custom_scores = cross_val_score(model, X, y, cv=kf)
print(f"Custom cross-validation scores: {custom_scores}")
print(f"Mean custom cross-validation score: {custom_scores.mean():.4f}")

# Perform cross-validation with Scikit-Learn's implementation
sklearn_kf = SklearnKFold(n_splits=5, shuffle=True, random_state=42)
sklearn_scores = sklearn_cross_val_score(model, X, y, cv=sklearn_kf)
print(f"Scikit-Learn cross-validation scores: {sklearn_scores}")
print(f"Mean Scikit-Learn cross-validation score: {sklearn_scores.mean():.4f}")
