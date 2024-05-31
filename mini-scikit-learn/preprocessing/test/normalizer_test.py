# File: preprocessing/test/normalizer_test.py

import unittest
import numpy as np
from sklearn.preprocessing import Normalizer as SklearnNormalizer
import sys
import os

# Ensure the parent directory is in the system path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.normalizer import Normalizer

class TestNormalizer(unittest.TestCase):
    def setUp(self):
        # Example data setup
        self.data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 0.0]
        ])

    def test_l2_normalizer(self):
        # Initialize both normalizers
        custom_normalizer = Normalizer(norm='l2')
        sklearn_normalizer = SklearnNormalizer(norm='l2')

        # Fit and transform the data
        custom_normalized = custom_normalizer.fit_transform(self.data)
        sklearn_normalized = sklearn_normalizer.fit_transform(self.data)

        # Check if both outputs are identical
        np.testing.assert_array_almost_equal(custom_normalized, sklearn_normalized, decimal=6)
        print("L2 Normalizer: OK" if np.allclose(custom_normalized, sklearn_normalized) else "L2 Normalizer: WRONG")

    def test_l1_normalizer(self):
        # Initialize both normalizers
        custom_normalizer = Normalizer(norm='l1')
        sklearn_normalizer = SklearnNormalizer(norm='l1')

        # Fit and transform the data
        custom_normalized = custom_normalizer.fit_transform(self.data)
        sklearn_normalized = sklearn_normalizer.fit_transform(self.data)

        # Check if both outputs are identical
        np.testing.assert_array_almost_equal(custom_normalized, sklearn_normalized, decimal=6)
        print("L1 Normalizer: OK" if np.allclose(custom_normalized, sklearn_normalized) else "L1 Normalizer: WRONG")

    def test_inplace_normalization(self):
        # Initialize both normalizers with inplace transformation
        custom_normalizer = Normalizer(norm='l2', copy=False)
        sklearn_normalizer = SklearnNormalizer(norm='l2', copy=False)

        # Fit and transform the data
        custom_normalized = custom_normalizer.fit_transform(self.data.copy())
        sklearn_normalized = sklearn_normalizer.fit_transform(self.data.copy())

        # Check if both outputs are identical
        np.testing.assert_array_almost_equal(custom_normalized, sklearn_normalized, decimal=6)
        print("Inplace Normalization: OK" if np.allclose(custom_normalized, sklearn_normalized) else "Inplace Normalization: WRONG")

if __name__ == '__main__':
    unittest.main()
