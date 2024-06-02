# GradientBoostingRegressor.py
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_model = None

    def fit(self, X, y):
        # Initialize the model list
        self.models = []
        
        # Initialize residuals
        residuals = y.copy()

        # Fit the initial model
        self.initial_model = np.mean(y)
        residuals -= self.initial_model

        for _ in range(self.n_estimators):
            # Fit a new model to the residuals
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, residuals)
            predictions = model.predict(X)
            
            # Update residuals
            residuals -= self.learning_rate * predictions
            
            # Store the model
            self.models.append(model)

        return self

    def predict(self, X):
        # Start with the initial model predictions
        predictions = np.full(X.shape[0], self.initial_model)
        
        # Add predictions from each tree
        for model in self.models:
            predictions += self.learning_rate * model.predict(X)
        
        return predictions

    def score(self, X, y):
        # Calculate R^2 score
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v
