
# Testing the implementation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.metrics import accuracy_score
from GaussianNB import GaussianNaiveBayes # type: ignore
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