# File: preprocessing/test/test_robust_scaler.py

import unittest
import numpy as np
from sklearn.preprocessing import RobustScaler as SklearnRobustScaler
import sys
import os

# Ensure the parent directory is in the system path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.robust_scaler import RobustScaler  # Ensure you have your implementation in the scalers module

class TestRobustScaler(unittest.TestCase):
    def test_robust_scaler(self):
        # Test data
        X = np.array([[1, 2], [3, 4], [5, 6]])

        # Our implementation
        scaler = RobustScaler()
        our_scaled = scaler.fit_transform(X)

        # Scikit-Learn implementation
        sk_scaler = SklearnRobustScaler()
        sk_scaled = sk_scaler.fit_transform(X)

        # Compare results
        np.testing.assert_almost_equal(our_scaled, sk_scaled)
        print("RobustScaler: OK" if np.allclose(our_scaled, sk_scaled) else "RobustScaler: WRONG")

if __name__ == '__main__':
    unittest.main()
