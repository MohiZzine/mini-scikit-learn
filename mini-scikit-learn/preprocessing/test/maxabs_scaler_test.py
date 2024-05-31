import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler as SklearnMaxAbsScaler

import sys
import os

# Ensure the parent directory is in the system path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from preprocessing.maxabs_scaler import MaxAbsScaler


def test_max_abs_scaler():
    # Test data
    X = np.array([
        [1., -1., 2.],
        [2., 0., 0.],
        [0., 1., -1.]
    ])

    # Custom implementation
    custom_scaler = MaxAbsScaler()
    custom_scaled = custom_scaler.fit_transform(X)

    # Sklearn implementation
    sklearn_scaler = SklearnMaxAbsScaler()
    sklearn_scaled = sklearn_scaler.fit_transform(X)

    # Print the results
    print("Original Data:\n", X)
    print("\nCustom MaxAbsScaler:\n", custom_scaled)
    print("\nSklearn MaxAbsScaler:\n", sklearn_scaled)

if __name__ == '__main__':
    test_max_abs_scaler()
    def __init__(self):
        self.data = np.array([
            [1., -1., 2.],
            [2., 0., 0.],
            [0., 1., -1.]
        ])

    def run_tests(self):
        self.plot_scaled_data()

    def plot_scaled_data(self):
        custom_scaler = MaxAbsScaler()
        sklearn_scaler = SklearnMaxAbsScaler()

        custom_scaled = custom_scaler.fit_transform(self.data)
        sklearn_scaled = sklearn_scaler.fit_transform(self.data)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(custom_scaled, cmap='viridis', interpolation='none')
        axes[0].set_title('Custom MaxAbsScaler')
        for (r, c), val in np.ndenumerate(custom_scaled):
            axes[0].text(c, r, f'{val:.2f}', ha='center', va='center', color='white')

        axes[1].imshow(sklearn_scaled, cmap='viridis', interpolation='none')
        axes[1].set_title('Sklearn MaxAbsScaler')
        for (r, c), val in np.ndenumerate(sklearn_scaled):
            axes[1].text(c, r, f'{val:.2f}', ha='center', va='center', color='white')

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    tester = test_max_abs_scaler()
