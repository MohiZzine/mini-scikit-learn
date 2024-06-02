from sklearn.datasets import load_wine
from sklearn.model_selection import  StratifiedKFold
from grid_search import GridSearchCV
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
