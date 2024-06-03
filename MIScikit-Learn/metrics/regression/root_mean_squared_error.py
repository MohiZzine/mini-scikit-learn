from mean_squared_error import mean_squared_error 
import numpy as np

def root_mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Compute root mean squared error regression loss.
    
    Args:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated target values.
    sample_weight (array-like, optional): Sample weights.
    multioutput (str or array-like): Aggregating of multiple output values.
    
    Returns:
    float or ndarray: Root mean squared error.
    """
    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
    rmse = np.sqrt(mse)
    return rmse


from sklearn.metrics import mean_squared_error as sklearn_mean_squared_error

# Test data
y_true_single = [3, -0.5, 2, 7]
y_pred_single = [2.5, 0.0, 2, 8]

# Test data for multiple outputs
y_true_multi = [[0.5, 1], [-1, 1], [7, -6]]
y_pred_multi = [[0, 2], [-1, 2], [8, -5]]

# Using our implementation
our_rmse_single = root_mean_squared_error(y_true_single, y_pred_single)
our_rmse_multi = root_mean_squared_error(y_true_multi, y_pred_multi)

print("Our RMSE (single):", our_rmse_single)
print("Our RMSE (multi):", our_rmse_multi)

# If sklearn's root_mean_squared_error exists, uncomment and compare
from sklearn.metrics import root_mean_squared_error as sklearn_root_mean_squared_error
sklearn_rmse_single = sklearn_root_mean_squared_error(y_true_single, y_pred_single)
sklearn_rmse_multi = sklearn_root_mean_squared_error(y_true_multi, y_pred_multi)

print("Sklearn RMSE (single):", sklearn_rmse_single)
print("Sklearn RMSE (multi):", sklearn_rmse_multi)
