
# Testing the updated implementation
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.metrics import accuracy_score
from KNN import KNeighborsClassifier
# Load the dataset (Iris and Wine datasets)
datasets = [load_iris, load_wine]

for dataset in datasets:
    data = dataset()
    X = data.data
    y = data.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Initialize and train our K-Nearest Neighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Predictions and accuracy
    y_pred = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)
    print(f"Custom K-Nearest Neighbors Accuracy on {dataset.__name__.replace('load_', '').capitalize()} dataset: {accuracy:.4f}")

    # Compare with Scikit-Learn's K-Nearest Neighbors
    sklearn_knn = SklearnKNeighborsClassifier(n_neighbors=5)
    sklearn_knn.fit(X_train, y_train)
    y_pred_sklearn = sklearn_knn.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Scikit-Learn K-Nearest Neighbors Accuracy on {dataset.__name__.replace('load_', '').capitalize()} dataset: {accuracy_sklearn:.4f}")


from collections import Counter
from sklearn.datasets import make_moons

# Create the make_moons dataset
X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train our K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions and accuracy
y_pred = knn.predict(X_test)
accuracy = knn.score(X_test, y_test)
print(f"Custom K-Nearest Neighbors Accuracy on make_moons dataset: {accuracy:.4f}")

# Compare with Scikit-Learn's K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
sklearn_knn = SklearnKNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(X_train, y_train)
y_pred_sklearn = sklearn_knn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Scikit-Learn K-Nearest Neighbors Accuracy on make_moons dataset: {accuracy_sklearn:.4f}")   

