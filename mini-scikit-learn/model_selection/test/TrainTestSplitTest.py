import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split as sk_train_test_split

# Ensure the parent directory is in the system path to allow package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model_selection.train_test_split import train_test_split

# Data setup
X = np.arange(10).reshape((5, 2))
y = np.arange(5)
test_size = 0.2
random_state = 42

# Using custom train_test_split
X_train_custom_1, X_test_custom_1, y_train_custom_1, y_test_custom_1 = train_test_split(
    X, y, test_size=test_size, random_state=random_state)

# Using custom train_test_split
X_train_custom_2, X_test_custom_2, y_train_custom_2, y_test_custom_2 = train_test_split(
    X, y, test_size=test_size, random_state=random_state)

# Using sklearn's train_test_split
X_train_sk, X_test_sk, y_train_sk, y_test_sk = sk_train_test_split(
    X, y, test_size=test_size, random_state=random_state)


# Printing results for manual comparison
print("Custom train_test_split results:")
print("X_train_custom_1:\n", X_train_custom_1)
print("X_test_custom_1:\n", X_test_custom_1)
print("y_train_custom_1:\n", y_train_custom_1)
print("y_test_custom_1:\n", y_test_custom_1)

print("\nCustom train_test_split results:")
print("X_train_custom_2:\n", X_train_custom_2)
print("X_test_custom_2:\n", X_test_custom_2)
print("y_train_custom_2:\n", y_train_custom_2)
print("y_test_custom_2:\n", y_test_custom_2)

print("\nSklearn train_test_split results:")
print("X_train_sk:\n", X_train_sk)
print("X_test_sk:\n", X_test_sk)
print("y_train_sk:\n", y_train_sk)
print("y_test_sk:\n", y_test_sk)


# # Printing results for manual comparison
# print("Custom train_test_split results:")
# print("X_train_custom:\n", X_train_custom)
# print("X_test_custom:\n", X_test_custom)
# print("y_train_custom:\n", y_train_custom)
# print("y_test_custom:\n", y_test_custom)

# print("\nSklearn train_test_split results:")
# print("X_train_sk:\n", X_train_sk)
# print("X_test_sk:\n", X_test_sk)
# print("y_train_sk:\n", y_train_sk)
# print("y_test_sk:\n", y_test_sk)

# # Test without shuffle
# X_train_custom_ns, X_test_custom_ns, y_train_custom_ns, y_test_custom_ns = train_test_split(
#     X, y, test_size=test_size, random_state=random_state, shuffle=False)

# X_train_sk_ns, X_test_sk_ns, y_train_sk_ns, y_test_sk_ns = sk_train_test_split(
#     X, y, test_size=test_size, random_state=random_state, shuffle=False)

# # Printing results for no shuffle case
# print("\nCustom train_test_split (no shuffle) results:")
# print("X_train_custom_ns:\n", X_train_custom_ns)
# print("X_test_custom_ns:\n", X_test_custom_ns)
# print("y_train_custom_ns:\n", y_train_custom_ns)
# print("y_test_custom_ns:\n", y_test_custom_ns)

# print("\nSklearn train_test_split (no shuffle) results:")
# print("X_train_sk_ns:\n", X_train_sk_ns)
# print("X_test_sk_ns:\n", X_test_sk_ns)
# print("y_train_sk_ns:\n", y_train_sk_ns)
# print("y_test_sk_ns:\n", y_test_sk_ns)

# # Test with stratify
# stratify = np.array([0, 0, 1, 1, 1])
# X_train_custom_str, X_test_custom_str, y_train_custom_str, y_test_custom_str = train_test_split(
#     X, y, test_size=test_size, random_state=random_state, stratify=stratify)

# X_train_sk_str, X_test_sk_str, y_train_sk_str, y_test_sk_str = sk_train_test_split(
#     X, y, test_size=test_size, random_state=random_state, stratify=stratify)

# # Printing results for stratified case
# print("\nCustom train_test_split (stratify) results:")
# print("X_train_custom_str:\n", X_train_custom_str)
# print("X_test_custom_str:\n", X_test_custom_str)
# print("y_train_custom_str:\n", y_train_custom_str)
# print("y_test_custom_str:\n", y_test_custom_str)

# print("\nSklearn train_test_split (stratify) results:")
# print("X_train_sk_str:\n", X_train_sk_str)
# print("X_test_sk_str:\n", X_test_sk_str)
# print("y_train_sk_str:\n", y_train_sk_str)
# print("y_test_sk_str:\n", y_test_sk_str)