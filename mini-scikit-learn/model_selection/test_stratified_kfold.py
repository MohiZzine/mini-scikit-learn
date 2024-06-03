import unittest
import numpy as np
from collections import Counter
from stratified_cross_validation import StratifiedKFold

class TestStratifiedKFold(unittest.TestCase):

    def setUp(self):
        from sklearn.datasets import load_iris
        self.X, self.y = load_iris(return_X_y=True)

    def test_split(self):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_indices = list(skf.split(self.X, self.y))
        
        self.assertEqual(len(fold_indices), 3)
        
        for train_idx, test_idx in fold_indices:
            self.assertEqual(len(train_idx) + len(test_idx), len(self.y))
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            train_counter = Counter(self.y[train_idx])
            test_counter = Counter(self.y[test_idx])
            for cls in np.unique(self.y):
                self.assertAlmostEqual(train_counter[cls] / len(train_idx), test_counter[cls] / len(test_idx), delta=0.1)

    def test_no_shuffle(self):
        skf = StratifiedKFold(n_splits=3, shuffle=False)
        fold_indices = list(skf.split(self.X, self.y))
        
        for train_idx, test_idx in fold_indices:
            self.assertEqual(len(train_idx) + len(test_idx), len(self.y))
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)

    def test_invalid_splits(self):
        with self.assertRaises(ValueError):
            StratifiedKFold(n_splits=1)

    def test_random_state(self):
        skf1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_indices1 = list(skf1.split(self.X, self.y))
        fold_indices2 = list(skf2.split(self.X, self.y))

        for (train_idx1, test_idx1), (train_idx2, test_idx2) in zip(fold_indices1, fold_indices2):
            np.testing.assert_array_equal(train_idx1, train_idx2)
            np.testing.assert_array_equal(test_idx1, test_idx2)

    def test_different_random_state(self):
        skf1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
        fold_indices1 = list(skf1.split(self.X, self.y))
        fold_indices2 = list(skf2.split(self.X, self.y))

        for (train_idx1, test_idx1), (train_idx2, test_idx2) in zip(fold_indices1, fold_indices2):
            self.assertFalse(np.array_equal(train_idx1, train_idx2))
            self.assertFalse(np.array_equal(test_idx1, test_idx2))

if __name__ == "__main__":
    unittest.main()
