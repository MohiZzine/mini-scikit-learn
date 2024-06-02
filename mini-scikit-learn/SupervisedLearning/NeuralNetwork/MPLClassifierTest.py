from MLPClassifier import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier as SKMLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom MLP Classifier
custom_clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', learning_rate=0.01, max_iter=500)
custom_clf.fit(X_train, y_train)
custom_pred = custom_clf.predict(X_test)

# Scikit-learn MLP Classifier
sk_clf = SKMLPClassifier(hidden_layer_sizes=(10,), activation='relu', learning_rate_init=0.01, max_iter=500)
sk_clf.fit(X_train, y_train)
sk_pred = sk_clf.predict(X_test)

# Compare accuracies
custom_accuracy = accuracy_score(y_test, custom_pred)
sk_accuracy = accuracy_score(y_test, sk_pred)

print("Custom MLP Classifier Accuracy:", custom_accuracy)
print("Scikit-learn MLP Classifier Accuracy:", sk_accuracy)


