import unittest
import numpy as np
from iterative_imputer import IterativeImputer

class TestIterativeImputer(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_incomplete = np.array([
            [1, 2, np.nan],
            [3, np.nan, 1],
            [np.nan, 6, 5],
            [8, 8, np.nan],
            [np.nan, 10, 9],
        ])
        self.X_complete = np.array([
            [1, 2, 3],
            [3, 4, 1],
            [5, 6, 5],
            [8, 8, 7],
            [6, 10, 9],
        ])

    def test_imputation(self):
        imputer = IterativeImputer(max_iter=10, random_state=42)
        X_imputed = imputer.fit_transform(self.X_incomplete)
        print(f"X_imputed:\n{X_imputed}\nExpected:\n{self.X_complete}")
        np.testing.assert_array_almost_equal(X_imputed, self.X_complete, decimal=0)

    def test_get_set_params(self):
        imputer = IterativeImputer(max_iter=10, random_state=42)
        params = imputer.get_params()
        self.assertEqual(params['max_iter'], 10)
        self.assertEqual(params['tol'], 1e-3)

        imputer.set_params(max_iter=20, tol=1e-4)
        self.assertEqual(imputer.max_iter, 20)
        self.assertEqual(imputer.tol, 1e-4)

if __name__ == "__main__":
    unittest.main()
