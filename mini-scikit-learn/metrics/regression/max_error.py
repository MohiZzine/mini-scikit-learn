import numpy as np

def max_error(y_true, y_pred):
    """
    Calculate the maximum residual error between true and predicted values.
    
    Args:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated target values.
    
    Returns:
    float: The maximum residual error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute the maximum error
    return np.max(np.abs(y_true - y_pred))

from sklearn.metrics import max_error as sklearn_max_error

# Test data
y_true = [3, 2, 7, 1]
y_pred = [4, 2, 7, 1]

# Using our implementation
our_max_error = max_error(y_true, y_pred)

# Using sklearn's implementation
sklearn_max_error_value = sklearn_max_error(y_true, y_pred)

print("Our Max Error:", our_max_error)
print("Sklearn Max Error:", sklearn_max_error_value)
