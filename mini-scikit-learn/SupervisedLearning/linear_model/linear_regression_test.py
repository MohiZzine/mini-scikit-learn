import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from linear_regression import LinearRegression


def test_linear_regression(n_samples=100, n_features=2, noise=10.0,
                           fit_intercept=True, positive=False):
    """
    Test the custom LinearRegression and compare it with scikit-learn's
    LinearRegression on generated regression data.

    Parameters
    ----------
    n_samples : int, optional, default: 100
        The number of samples.
    n_features : int, optional, default: 2
        The number of features.
    noise : float, optional, default: 10.0
        The standard deviation of the gaussian noise applied to the output.
    fit_intercept : bool, optional, default: True
        Whether to calculate the intercept for this model.
    positive : bool, optional, default: False
        When set to True, forces the coefficients to be positive.
    """
    # Generate random data
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           noise=noise, coef=False, random_state=42)

    if positive:
        # Ensure all coefficients and dependent variable are positive
        X = np.abs(X)
        y = np.abs(y)

    # Custom Linear Regression
    custom_model = LinearRegression(fit_intercept=fit_intercept,
                                    positive=positive)
    custom_model.fit(X, y)
    custom_predictions = custom_model.predict(X)
    custom_score = r2_score(y, custom_predictions)

    # Sklearn Linear Regression
    sklearn_model = SklearnLinearRegression(fit_intercept=fit_intercept,
                                            positive=positive)
    sklearn_model.fit(X, y)
    sklearn_predictions = sklearn_model.predict(X)
    sklearn_score = r2_score(y, sklearn_predictions)

    print(f"Testing dataset with {n_samples} samples and {n_features} features:")
    print(f"Custom Model R^2 Score: {custom_score:.4f}")
    print(f"Scikit-Learn R^2 Score: {sklearn_score:.4f}")
    print("Difference in R^2 scores:", abs(custom_score - sklearn_score))
    print("---" * 10)


# Test cases
test_linear_regression(n_samples=100, n_features=2, noise=5.0,
                       fit_intercept=True, positive=False)
test_linear_regression(n_samples=200, n_features=10, noise=10.0,
                       fit_intercept=True, positive=True)
test_linear_regression(n_samples=150, n_features=5, noise=20.0,
                       fit_intercept=False, positive=False)
