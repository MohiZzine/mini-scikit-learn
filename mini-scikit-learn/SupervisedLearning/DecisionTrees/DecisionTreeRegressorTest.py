from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from DecisionTreeRegressor import DecisionTreeRegressor


X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train our Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=3)
dt.fit(X_train, y_train)

# Predictions and evaluation
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print(f"Custom Decision Tree Regressor Train MSE: {mse_train:.4f}")
print(f"Custom Decision Tree Regressor Test MSE: {mse_test:.4f}")

# Compare with Scikit-Learn's Decision Tree Regressor
sklearn_dt = SklearnDecisionTreeRegressor(max_depth=3)
sklearn_dt.fit(X_train, y_train)
y_pred_sklearn_train = sklearn_dt.predict(X_train)
y_pred_sklearn_test = sklearn_dt.predict(X_test)
mse_sklearn_train = mean_squared_error(y_train, y_pred_sklearn_train)
mse_sklearn_test = mean_squared_error(y_test, y_pred_sklearn_test)
print(f"Scikit-Learn Decision Tree Regressor Train MSE: {mse_sklearn_train:.4f}")
print(f"Scikit-Learn Decision Tree Regressor Test MSE: {mse_sklearn_test:.4f}")

