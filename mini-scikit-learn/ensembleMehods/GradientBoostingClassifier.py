from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.scaling_factor = 1.0

    def fit(self, X, y):
        self.models = []
        residual = np.zeros_like(y, dtype=float)
        for _ in range(self.n_estimators):
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residual)
            self.models.append(model)
            update = self.learning_rate * model.predict(X)
            residual -= update

    def predict(self, X):
        predictions = np.sum([self.learning_rate * model.predict(X) for model in self.models], axis=0)
        return np.round(predictions).astype(int)


# Testing with Iris Dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom Gradient Boosting
model_custom = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model_custom.fit(X_train, y_train)
predictions_custom = model_custom.predict(X_test)
accuracy_custom = accuracy_score(y_test, predictions_custom)

# Scikit-learn Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier as SklearnGradientBoostingClassifier
model_sklearn = SklearnGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
model_sklearn.fit(X_train, y_train)
predictions_sklearn = model_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)

print(f"Custom GradientBoostingClassifier Accuracy:", accuracy_custom)
print(f"Scikit-learn GradientBoostingClassifier Accuracy:", accuracy_sklearn)
