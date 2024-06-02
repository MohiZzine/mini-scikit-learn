# supervised_learning/mlp_classifier/mlp_classifier.py

import numpy as np

class MLPClassifier:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001, max_iter=200, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.activation_function = self._get_activation_function(activation)
        self.activation_derivative = self._get_activation_derivative(activation)

    def _get_activation_function(self, activation):
        if activation == 'relu':
            return lambda x: np.maximum(0, x)
        elif activation == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def _get_activation_derivative(self, activation):
        if activation == 'relu':
            return lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            return lambda x: sigmoid(x) * (1 - sigmoid(x))
        elif activation == 'tanh':
            return lambda x: 1 - np.tanh(x)**2
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def _initialize_weights(self, input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)

    def fit(self, X, y):
        np.random.seed(42)
        n_samples, n_features = X.shape
        n_outputs = len(np.unique(y))

        # One-hot encode the output labels
        y_encoded = np.eye(n_outputs)[y]

        # Initialize weights, biases, and Adam parameters
        self.weights = []
        self.biases = []
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []

        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]

        for i in range(len(layer_sizes) - 1):
            self.weights.append(self._initialize_weights(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
            self.m_weights.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
            self.v_weights.append(np.zeros((layer_sizes[i], layer_sizes[i + 1])))
            self.m_biases.append(np.zeros((1, layer_sizes[i + 1])))
            self.v_biases.append(np.zeros((1, layer_sizes[i + 1])))

        for t in range(1, self.max_iter + 1):
            # Forward pass
            activations = [X]
            pre_activations = []

            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                Z = np.dot(activations[-1], W) + b
                pre_activations.append(Z)
                activations.append(self.activation_function(Z))

            Z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
            pre_activations.append(Z)
            activations.append(np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True))

            # Backward pass
            delta = activations[-1] - y_encoded

            for i in reversed(range(len(self.weights))):
                dW = np.dot(activations[i].T, delta)
                db = np.sum(delta, axis=0, keepdims=True)

                if i > 0:
                    delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(pre_activations[i - 1])

                # Adam parameter updates
                self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * dW
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dW ** 2)
                m_hat_weights = self.m_weights[i] / (1 - self.beta1 ** t)
                v_hat_weights = self.v_weights[i] / (1 - self.beta2 ** t)

                self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * db
                self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (db ** 2)
                m_hat_biases = self.m_biases[i] / (1 - self.beta1 ** t)
                v_hat_biases = self.v_biases[i] / (1 - self.beta2 ** t)

                # Update weights and biases
                self.weights[i] -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

    def predict(self, X):
        activations = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            activations = self.activation_function(np.dot(activations, W) + b)
        Z = np.dot(activations, self.weights[-1]) + self.biases[-1]
        probabilities = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        return np.argmax(probabilities, axis=1)


# Additional functions for evaluation and comparison
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def compare_mlp():
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

    
    
if __name__ == "__main__":
    compare_mlp()