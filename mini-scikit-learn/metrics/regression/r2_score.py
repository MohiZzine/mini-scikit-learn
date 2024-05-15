import numpy as np

def r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average', force_finite=True):
    """
    Compute R^2 (coefficient of determination) regression score function.
    
    Args:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated target values.
    sample_weight (array-like, optional): Sample weights.
    multioutput (str or array-like): Defines how to aggregate multiple output scores.
    force_finite (bool, optional): Adjust non-finite scores for constant target data.
    
    Returns:
    float or ndarray: The R^2 score or ndarray of scores if multioutput is 'raw_values'.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Check for single or multioutput data
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    if sample_weight is not None:
        sample_weight = np.array(sample_weight)

    # Calculate the Total Sum of Squares (TSS)
    weight_sum = np.sum(sample_weight) if sample_weight is not None else y_true.shape[0]
    y_true_mean = np.average(y_true, weights=sample_weight, axis=0)
    total_sum_squares = np.sum((sample_weight * (y_true - y_true_mean) ** 2 if sample_weight is not None else (y_true - y_true_mean) ** 2), axis=0)

    # Calculate the Residual Sum of Squares (RSS)
    residual_sum_squares = np.sum((sample_weight * (y_true - y_pred) ** 2 if sample_weight is not None else (y_true - y_pred) ** 2), axis=0)

    # Calculate R2
    r2 = 1 - (residual_sum_squares / total_sum_squares)

    # Handling non-finite values
    if force_finite:
        r2[np.isinf(r2) | np.isnan(r2)] = 0.0

    # Handling multioutput options
    if multioutput == 'variance_weighted':
        if total_sum_squares.sum() == 0:
            return 0.0 if any(residual_sum_squares) else 1.0
        return np.average(r2, weights=total_sum_squares)
    elif multioutput == 'raw_values':
        return r2
    elif isinstance(multioutput, (list, np.ndarray)):
        return np.average(r2, weights=multioutput)
    else:  # Default 'uniform_average'
        return np.mean(r2)

# Example usage
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print("R2 Score:", r2_score(y_true, y_pred))

y_true_multi = [[0.5, 1], [-1, 1], [7, -6]]
y_pred_multi = [[0, 2], [-1, 2], [8, -5]]
print("R2 Score, Multioutput Variance-Weighted:", r2_score(y_true_multi, y_pred_multi, multioutput='variance_weighted'))


from sklearn.metrics import r2_score as sk_r2_score

print("R2 Score Sklearn:", sk_r2_score(y_true, y_pred))
print("R2 Score Sklearn, Multioutput Variance-Weighted:", sk_r2_score(y_true_multi, y_pred_multi, multioutput='variance_weighted'))

