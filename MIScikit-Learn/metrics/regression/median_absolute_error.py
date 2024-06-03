import numpy as np

def median_absolute_error(y_true, y_pred, multioutput='uniform_average', sample_weight=None):
    """
    Compute median absolute error regression loss.
    
    Args:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated target values.
    multioutput (str or array-like): Defines aggregating of multiple output values.
    sample_weight (array-like, optional): Sample weights.
    
    Returns:
    float or ndarray: Median absolute error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    errors = np.abs(y_true - y_pred)

    if sample_weight is not None:
        errors *= np.array(sample_weight)[:, np.newaxis]

    if multioutput == 'raw_values':
        return np.median(errors, axis=0)
    elif isinstance(multioutput, (list, np.ndarray)):
        weights = np.array(multioutput)
        return np.average(np.median(errors, axis=0), weights=weights)
    else:  # 'uniform_average'
        return np.median(errors)

# Example usage and testing
y_true_single = [3, -0.5, 2, 7]
y_pred_single = [2.5, 0.0, 2, 8]

y_true_multi = [[0.5, 1], [-1, 1], [7, -6]]
y_pred_multi = [[0, 2], [-1, 2], [8, -5]]

mae_single = median_absolute_error(y_true_single, y_pred_single)
mae_multi = median_absolute_error(y_true_multi, y_pred_multi)
mae_multi_raw = median_absolute_error(y_true_multi, y_pred_multi, multioutput='raw_values')
mae_multi_weighted = median_absolute_error(y_true_multi, y_pred_multi, multioutput=[0.3, 0.7])

print("Median Absolute Error (single):", mae_single)
print("Median Absolute Error (multi):", mae_multi)
print("Median Absolute Error (multi, raw):", mae_multi_raw)
print("Median Absolute Error (multi, weighted):", mae_multi_weighted)

from sklearn.metrics import median_absolute_error as sk_median_absolute_error

sk_mae_single = sk_median_absolute_error(y_true_single, y_pred_single)
sk_mae_multi = sk_median_absolute_error(y_true_multi, y_pred_multi)
sk_mae_multi_raw = sk_median_absolute_error(y_true_multi, y_pred_multi, multioutput='raw_values')
sk_mae_multi_weighted = sk_median_absolute_error(y_true_multi, y_pred_multi, multioutput=[0.3, 0.7])

print("\nScikit-learn:")
print("Median Absolute Error (single):", sk_mae_single)
print("Median Absolute Error (multi):", sk_mae_multi)
print("Median Absolute Error (multi, raw):", sk_mae_multi_raw)
print("Median Absolute Error (multi, weighted):", sk_mae_multi_weighted)

