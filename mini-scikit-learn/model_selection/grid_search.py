import numpy as np
from itertools import product
from metrics.classification.accuracy import accuracy_score
from metrics.classification.precision_score import precision_score
from metrics.classification.f1_score import f1_score
from metrics.classification.recall_score import recall_score
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold

class GridSearchCV:
    def __init__(self, estimator, param_grid, scoring='accuracy', cv=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def fit(self, X, y):
        param_grid = list(ParameterGrid(self.param_grid))
        results = []
        best_score = -np.inf
        
        for params in param_grid:
            self.estimator.set_params(**params)
            scores = cross_val_score(self.estimator, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = np.mean(scores)
            results.append((params, mean_score))

            if mean_score > best_score:
                best_score = mean_score
                self.best_params_ = params
                self.best_score_ = best_score
                self.best_estimator_ = clone(self.estimator).set_params(**params)
        
        self.cv_results_ = results
        self.best_estimator_.fit(X, y)

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)

class ParameterGrid:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        keys, values = zip(*self.param_grid.items())
        for v in product(*values):
            params = dict(zip(keys, v))
            yield params

def cross_val_score(model, X, y, cv, scoring='accuracy'):
    if scoring == 'accuracy':
        scoring_func = accuracy_score
    elif scoring == 'precision':
        scoring_func = precision_score
    elif scoring == 'recall':
        scoring_func = recall_score
    elif scoring == 'f1':
        scoring_func = f1_score
    else:
        raise ValueError("Unsupported scoring method.")

    scores = []
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scoring_func(y_test, y_pred)
        scores.append(score)
    return np.array(scores)

def clone(estimator):
    estimator_clone = estimator.__class__()
    estimator_clone.__dict__.update(estimator.__dict__)
    return estimator_clone

# Testing the implementation
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold as SklearnStratifiedKFold
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Parameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

# Initialize the custom GridSearchCV
cv = SklearnStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Compare with Scikit-Learn's GridSearchCV
sklearn_grid_search = SklearnGridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')
sklearn_grid_search.fit(X, y)

print(f"Scikit-Learn Best parameters: {sklearn_grid_search.best_params_}")
print(f"Scikit-Learn Best cross-validation score: {sklearn_grid_search.best_score_:.4f}")


# Load the Wine dataset
data = load_wine()
X, y = data.data, data.target

# Initialize the model
model = LogisticRegression(max_iter=10000)

# Parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga']
}

# Initialize the custom GridSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Compare with Scikit-Learn's GridSearchCV
from sklearn.model_selection import GridSearchCV as SklearnGridSearchCV

sklearn_grid_search = SklearnGridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy')
sklearn_grid_search.fit(X, y)

print(f"Scikit-Learn Best parameters: {sklearn_grid_search.best_params_}")
print(f"Scikit-Learn Best cross-validation score: {sklearn_grid_search.best_score_:.4f}")
