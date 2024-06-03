import unittest
import numpy as np
from sklearn.neural_network import MLPRegressor
from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # XOR problem
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])
        
        # Custom Neural Network
        self.nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)
        self.nn.fit(self.X, self.y, epochs=15000)
        
        # Scikit-Learn Neural Network
        self.sklearn_nn = MLPRegressor(hidden_layer_sizes=(2,), max_iter=15000, learning_rate_init=0.1, solver='sgd', activation='logistic', random_state=42)
        self.sklearn_nn.fit(self.X, self.y.ravel())

    def test_predictions(self):
        custom_predictions = self.nn.predict(self.X)
        sklearn_predictions = self.sklearn_nn.predict(self.X)
        
        # Verify that predictions are similar (within a tolerance)
        np.testing.assert_allclose(custom_predictions.ravel(), sklearn_predictions, rtol=0.5, atol=0.5)

if __name__ == "__main__":
    unittest.main()
