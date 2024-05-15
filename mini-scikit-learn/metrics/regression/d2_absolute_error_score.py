import numpy as np

def d2_absolute_error_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Compute the d2_absolute_error_score.

    Args:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated target values.
    sample_weight (array-like, optional): Sample weights.
    multioutput (str or array-like): Defines how to aggregate multiple output scores.

    Returns:
    float or ndarray: The d2 absolute error score or ndarray of scores if `multioutput` is 'raw_values'.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    # Calculate the absolute errors
    abs_errors = np.abs(y_true - y_pred)
    median_errors = np.abs(y_true - np.median(y_true, axis=0))

    if sample_weight is not None:
        sample_weight = np.array(sample_weight).reshape(-1, 1)
        abs_errors *= sample_weight
        median_errors *= sample_weight

    # Sum of errors
    model_error = np.sum(abs_errors, axis=0)
    median_error = np.sum(median_errors, axis=0)

    # Calculate d2 score
    scores = 1 - (model_error / median_error)

    # Handling multioutput options
    if multioutput == 'raw_values':
        return scores
    elif isinstance(multioutput, (list, np.ndarray)):
        weights = np.array(multioutput)
        return np.average(scores, weights=weights)
    else:  # Default 'uniform_average'
        return np.mean(scores)

# Example usage and testing
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print("D2 Absolute Error Score:", d2_absolute_error_score(y_true, y_pred))

y_true_multi = [[0.5, 1], [-1, 1], [7, -6]]
y_pred_multi = [[0, 2], [-1, 2], [8, -5]]
print("D2 Absolute Error Score, Multioutput Uniform Average:", d2_absolute_error_score(y_true_multi, y_pred_multi))
print("D2 Absolute Error Score, Multioutput Raw Values:", d2_absolute_error_score(y_true_multi, y_pred_multi, multioutput='raw_values'))


from sklearn.metrics import sklearn_d2_absolute_error_score

# Test the implementation against the sklearn implementation
assert np.allclose(d2_absolute_error_score(y_true, y_pred), sklearn_d2_absolute_error_score(y_true, y_pred))
assert np.allclose(d2_absolute_error_score(y_true_multi, y_pred_multi), sklearn_d2_absolute_error_score(y_true_multi, y_pred_multi))
assert np.allclose(d2_absolute_error_score(y_true_multi, y_pred_multi, multioutput='raw_values'), sklearn_d2_absolute_error_score(y_true_multi, y_pred_multi, multioutput='raw_values'))
print("Implementation is correct!")