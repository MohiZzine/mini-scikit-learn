import numpy as np

class MLPClassifier:
    """
    Multi-layer Perceptron (MLP) classifier.

    Parameters
    ----------
    hidden_layer_sizes : tuple, optional, default: (100,)
        The ith element represents the number of neurons in the ith hidden layer.
    activation : {'relu', 'sigmoid', 'tanh'}, optional, default: 'relu'
        Activation function for the hidden layer.
    learning_rate : float, optional, default: 0.001
        Learning rate for weight updates.
    max_iter : int, optional, default: 200
        Maximum number of iterations.
    beta1 : float, optional, default: 0.9
        Exponential decay rate for the first moment estimates in Adam optimizer.
    beta2 : float, optional, default: 0.999
        Exponential decay rate for the second moment estimates in Adam optimizer.
    epsilon : float, optional, default: 1e-8
        Value to avoid division by zero in Adam optimizer.
    """

    def __init__(
        self, hidden_layer_sizes=(100,), activation='relu', learning_rate=0.001,
        max_iter=200, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
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
        """
        Get the activation function.

        Parameters
        ----------
        activation : str
            Activation function name.

        Returns
        -------
        function
            Corresponding activation function.
        """
        if activation == 'relu':
            return lambda x: np.maximum(0, x)
        elif activation == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif activation == 'tanh':
            return lambda x: np.tanh(x)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def _get_activation_derivative(self, activation):
        """
        Get the derivative of the activation function.

        Parameters
        ----------
        activation : str
            Activation function name.

        Returns
        -------
        function
            Corresponding derivative of the activation function.
        """
        if activation == 'relu':
            return lambda x: (x > 0).astype(float)
        elif activation == 'sigmoid':
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            return lambda x: sigmoid(x) * (1 - sigmoid(x))
        elif activation == 'tanh':
            return lambda x: 1 - np.tanh(x) ** 2
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def _initialize_weights(self, input_size, output_size):
        """
        Initialize weights for the layer.

        Parameters
        ----------
        input_size : int
            Size of input layer.
        output_size : int
            Size of output layer.

        Returns
        -------
        np.array
            Initialized weights.
        """
        return np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)

    def fit(self, X, y):
        """
        Fit the MLP model to the data.

        Parameters
        ----------
        X : np.array
            Training data.
        y : np.array
            Target values.
        """
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
                self.weights[i] -= self.learning_rate * m_hat_weights / (
                    np.sqrt(v_hat_weights) + self.epsilon
                )
                self.biases[i] -= self.learning_rate * m_hat_biases / (
                    np.sqrt(v_hat_biases) + self.epsilon
                )

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.array
            Input data.

        Returns
        -------
        np.array
            Predicted class labels.
        """
        activations = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            activations = self.activation_function(np.dot(activations, W) + b)
        Z = np.dot(activations, self.weights[-1]) + self.biases[-1]
        probabilities = np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)
        return np.argmax(probabilities, axis=1)
