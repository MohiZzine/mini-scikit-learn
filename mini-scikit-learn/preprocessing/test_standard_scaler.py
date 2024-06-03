import unittest
import numpy as np
from standard_scaler import StandardScaler

class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])
        self.scaler = StandardScaler()

    def test_fit(self):
        self.scaler.fit(self.X)
        expected_mean = [7, 8, 9]
        expected_scale = [4.24264069, 4.24264069, 4.24264069]
        np.testing.assert_almost_equal(self.scaler.mean_, expected_mean, decimal=1)
        np.testing.assert_almost_equal(self.scaler.scale_, expected_scale, decimal=1)

    def test_transform(self):
        self.scaler.fit(self.X)
        X_trans = self.scaler.transform(self.X)
        expected_trans = np.array([
            [-1.4, -1.4, -1.4],
            [-0.7, -0.7, -0.7],
            [0, 0, 0],
            [0.7, 0.7, 0.7],
            [1.4, 1.4, 1.4]
        ])
        np.testing.assert_almost_equal(X_trans, expected_trans, decimal=1)

    def test_fit_transform(self):
        X_trans = self.scaler.fit_transform(self.X)
        expected_trans = np.array([
            [-1.4, -1.4, -1.4],
            [-0.7, -0.7, -0.7],
            [0, 0, 0],
            [0.7, 0.7, 0.7],
            [1.4, 1.4, 1.4]
        ])
        np.testing.assert_almost_equal(X_trans, expected_trans, decimal=1)

    def test_inverse_transform(self):
        self.scaler.fit(self.X)
        X_trans = self.scaler.transform(self.X)
        X_inv = self.scaler.inverse_transform(X_trans)
        np.testing.assert_almost_equal(self.X, X_inv, decimal=1)

    def test_get_set_params(self):
        params = self.scaler.get_params()
        self.assertTrue(params['with_mean'])
        self.assertTrue(params['with_std'])
        self.scaler.set_params(with_mean=False)
        self.assertFalse(self.scaler.with_mean)

    def test_partial_fit(self):
        X1 = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])
        X2 = np.array([
            [7, 8, 9],
            [10, 11, 12],
            [13, 14, 15]
        ])
        self.scaler.partial_fit(X1)
        self.scaler.partial_fit(X2)
        expected_mean = [7, 8, 9]
        expected_scale = [4.24264069, 4.24264069, 4.24264069]
        np.testing.assert_almost_equal(self.scaler.mean_, expected_mean, decimal=1)
        np.testing.assert_almost_equal(self.scaler.scale_, expected_scale, decimal=1)

if __name__ == "__main__":
    unittest.main()
