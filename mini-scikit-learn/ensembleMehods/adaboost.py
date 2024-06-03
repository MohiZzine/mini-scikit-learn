import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)

            error = np.sum(w * (y_pred != y)) / np.sum(w)
            alpha = self.learning_rate * np.log((1 - error) / (error + 1e-10)) / 2

            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for model, alpha in zip(self.models, self.alphas):
            y_pred += alpha * model.predict(X)
        return np.sign(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
