import numpy as np

class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.means = None
        self.variances = None
        self.priors = None

    def fit(self, X, y):
        # Determine the unique classes
        self.classes = np.unique(y)
        # Initialize the means, variances, and priors
        self.means = {}
        self.variances = {}
        self.priors = {}
        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.variances[cls] = np.var(X_cls, axis=0)
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def _calculate_likelihood(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.variances[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def _calculate_posterior(self, x):
        posteriors = []
        for cls in self.classes:
            prior = np.log(self.priors[cls])
            likelihood = np.sum(np.log(self._calculate_likelihood(cls, x)))
            posterior = prior + likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._calculate_posterior(x) for x in X]
        return np.array(y_pred)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Testing the implementation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.metrics import accuracy_score
def test_GaussianNaiveBayes():  
    # Load the dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Initialize and train our Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    # Predictions and accuracy
    y_pred = gnb.predict(X_test)
    accuracy = gnb.score(X_test, y_test)
    print(f"Custom Gaussian Naive Bayes Accuracy: {accuracy:.4f}")

    # Compare with Scikit-Learn's Gaussian Naive Bayes
    sklearn_gnb = SklearnGaussianNB()
    sklearn_gnb.fit(X_train, y_train)
    y_pred_sklearn = sklearn_gnb.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Scikit-Learn Gaussian Naive Bayes Accuracy: {accuracy_sklearn:.4f}")

test_GaussianNaiveBayes()

from sklearn.datasets import load_wine

def test2_GaussianNaiveBayes():
    # Load the dataset
    data = load_wine()
    X = data.data
    y = data.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Initialize and train our Gaussian Naive Bayes
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train, y_train)

    # Predictions and accuracy
    y_pred = gnb.predict(X_test)
    accuracy = gnb.score(X_test, y_test)
    print(f"Custom Gaussian Naive Bayes Accuracy: {accuracy:.4f}")

    # Compare with Scikit-Learn's Gaussian Naive Bayes
    sklearn_gnb = SklearnGaussianNB()
    sklearn_gnb.fit(X_train, y_train)
    y_pred_sklearn = sklearn_gnb.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Scikit-Learn Gaussian Naive Bayes Accuracy: {accuracy_sklearn:.4f}")                

test2_GaussianNaiveBayes()