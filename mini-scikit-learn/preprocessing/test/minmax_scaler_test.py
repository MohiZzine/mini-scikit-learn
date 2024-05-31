# File: preprocessing/test/test_min_max_scaler.py

import unittest
import numpy as np
from sklearn.preprocessing import MinMaxScaler as SklearnMinMaxScaler
import sys
import os

# Ensure the parent directory is in the system path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.minmax_scaler import MinMaxScaler

class TestMinMaxScaler(unittest.TestCase):
    def test_min_max_scaler(self):
        # Test data
        X = np.array([[1, 2], [3, 4], [5, 6]])

        # Our implementation
        scaler = MinMaxScaler()
        our_scaled = scaler.fit_transform(X)

        # Scikit-Learn implementation
        sk_scaler = SklearnMinMaxScaler()
        sk_scaled = sk_scaler.fit_transform(X)

        # Compare results
        np.testing.assert_almost_equal(our_scaled, sk_scaled)
        print("MinMaxScaler: OK" if np.allclose(our_scaled, sk_scaled) else "MinMaxScaler: WRONG")

if __name__ == '__main__':
    unittest.main()
