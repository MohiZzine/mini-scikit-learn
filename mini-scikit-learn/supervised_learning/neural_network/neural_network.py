import numpy as np
# from ..base import BaseEstimator
from sklearn.neural_network import MLPRegressor

# Activation function: Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Mean Squared Error Loss Function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1 / self.input_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1 / self.hidden_size)
        
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output

    def backward(self, X, y):
        loss = mse_loss(y, self.final_output)
        
        output_error = self.final_output - y
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        self.weights_hidden_output -= self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        
        self.weights_input_hidden -= X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        
        return loss

    def fit(self, X, y, epochs=15000):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y)
            if epoch % 1000 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)

# Example usage and comparison with Scikit-Learn's neural network
if __name__ == "__main__":
    # XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Custom Neural Network
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
    nn.fit(X, y, epochs=15000)
    custom_predictions = nn.predict(X)
    print("Custom Neural Network Predictions:")
    print(custom_predictions)
    
    # Scikit-Learn Neural Network
    sklearn_nn = MLPRegressor(hidden_layer_sizes=(2,), max_iter=15000, learning_rate_init=0.1, solver='sgd', activation='logistic', random_state=42)
    sklearn_nn.fit(X, y.ravel())
    sklearn_predictions = sklearn_nn.predict(X)
    print("Scikit-Learn Neural Network Predictions:")
    print(sklearn_predictions)
