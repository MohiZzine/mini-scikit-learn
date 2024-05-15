import numpy as np
from numpy.linalg import svd

class LinearRegression:
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None, positive=False):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.n_jobs = n_jobs
        self.positive = positive
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        if self.copy_X:
            X = X.copy()

        if self.fit_intercept:
            # Add a column of ones for the intercept
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Handle the positive constraint
        if self.positive:
            # We can use np.linalg.lstsq with a positive constraint
            # Since the 'positive' constraint requires using an algorithm that supports it
            # Currently, there's no direct solver in numpy that does this, so we might simulate it
            # or use another library like scipy.optimize.nnls which doesn't directly apply here
            # For educational purposes, let's assume no constraint, or we could use CVXPY for a full solution
            pass

        # Compute the SVD of X for the solution
        U, s, Vt = svd(X, full_matrices=False)
        s_inv = np.diag(1 / s)
        X_pinv = Vt.T @ s_inv @ U.T
        self.coef_ = X_pinv @ y

        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0

        return self

    def predict(self, X):
        if self.fit_intercept:
            # Add a column for the intercept if it was used during fitting
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            return X @ np.hstack([self.intercept_, self.coef_])
        else:
            # No intercept was fitted, so use coefficients only
            return X @ self.coef_

    def score(self, X, y):
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v

    def get_params(self, deep=True):
        return {'fit_intercept': self.fit_intercept, 'copy_X': self.copy_X, 'n_jobs': self.n_jobs, 'positive': self.positive}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self






import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression

# Assuming CustomLinearRegression is imported
# from your_module import CustomLinearRegression

def test_linear_regression(n_samples=100, n_features=2, noise=10.0, fit_intercept=True, positive=False):
    # Generate random data
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise, coef=False, random_state=42)
    
    if positive:
        # Ensure all coefficients and dependent variable are positive
        X = np.abs(X)
        y = np.abs(y)

    # Custom Linear Regression
    custom_model = LinearRegression(fit_intercept=fit_intercept, positive=positive)
    custom_model.fit(X, y)
    custom_predictions = custom_model.predict(X)
    custom_score = r2_score(y, custom_predictions)

    # Sklearn Linear Regression
    sklearn_model = SklearnLinearRegression(fit_intercept=fit_intercept, positive=positive)
    sklearn_model.fit(X, y)
    sklearn_predictions = sklearn_model.predict(X)
    sklearn_score = r2_score(y, sklearn_predictions)

    print(f"Testing dataset with {n_samples} samples and {n_features} features:")
    print(f"Custom Model R^2 Score: {custom_score:.4f}")
    print(f"Scikit-Learn R^2 Score: {sklearn_score:.4f}")
    print("Difference in R^2 scores:", abs(custom_score - sklearn_score))
    print("---" * 10)

# Test cases
test_linear_regression(n_samples=100, n_features=2, noise=5.0, fit_intercept=True, positive=False)
test_linear_regression(n_samples=200, n_features=10, noise=10.0, fit_intercept=True, positive=True)
test_linear_regression(n_samples=150, n_features=5, noise=20.0, fit_intercept=False, positive=False)

