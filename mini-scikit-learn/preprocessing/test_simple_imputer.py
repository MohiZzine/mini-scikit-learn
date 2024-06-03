import unittest
import numpy as np
from simple_imputer import SimpleImputer

class TestSimpleImputer(unittest.TestCase):
    def setUp(self):
        self.X_incomplete = np.array([
            [1, 2, np.nan],
            [3, np.nan, 1],
            [np.nan, 6, 5],
            [8, 8, np.nan],
            [np.nan, 10, 9],
        ])
        self.expected_mean = np.array([
            [1, 2, 5],
            [3, 6.5, 1],
            [4, 6, 5],
            [8, 8, 5],
            [4, 10, 9],
        ])
        self.expected_median = np.array([
            [1, 2, 5],
            [3, 7, 1],
            [4, 6, 5],
            [8, 8, 5],
            [4, 10, 9],
        ])
        self.expected_most_frequent = np.array([
            [1, 2, 1],
            [3, 6, 1],
            [1, 6, 5],
            [8, 8, 1],
            [1, 10, 9],
        ])
        self.expected_constant = np.array([
            [1, 2, -1],
            [3, -1, 1],
            [-1, 6, 5],
            [8, 8, -1],
            [-1, 10, 9],
        ])

    def test_imputation_mean(self):
        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(self.X_incomplete)
        np.testing.assert_array_almost_equal(X_imputed, self.expected_mean)

    def test_imputation_median(self):
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(self.X_incomplete)
        np.testing.assert_array_almost_equal(X_imputed, self.expected_median)

    def test_imputation_most_frequent(self):
        imputer = SimpleImputer(strategy="most_frequent")
        X_imputed = imputer.fit_transform(self.X_incomplete)
        np.testing.assert_array_almost_equal(X_imputed, self.expected_most_frequent)

    def test_imputation_constant(self):
        imputer = SimpleImputer(strategy="constant", fill_value=-1)
        X_imputed = imputer.fit_transform(self.X_incomplete)
        np.testing.assert_array_almost_equal(X_imputed, self.expected_constant)

    def test_get_set_params(self):
        imputer = SimpleImputer(strategy="mean")
        params = imputer.get_params()
        self.assertEqual(params['strategy'], "mean")
        self.assertEqual(params['fill_value'], None)

        imputer.set_params(strategy="median", fill_value=0)
        self.assertEqual(imputer.strategy, "median")
        self.assertEqual(imputer.fill_value, 0)

if __name__ == "__main__":
    unittest.main()