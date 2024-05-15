import numpy as np

def mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Calculate the Mean Absolute Error between true and predicted values.

    Args:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    sample_weight (array-like, optional): Sample weights.
    multioutput (str or array-like, optional): Defines aggregating of multiple output values.

    Returns:
    float or ndarray: The mean absolute error or array of mean absolute errors for each output.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the absolute errors
    errors = np.abs(y_pred - y_true)

    # Apply sample weights if provided
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        errors *= sample_weight

    # Handle multioutput aggregation
    if multioutput == 'raw_values':
        # Return the mean error for each output separately
        result = np.mean(errors, axis=0)
    elif isinstance(multioutput, (list, np.ndarray)):
        # Compute the weighted average of all output errors
        multioutput = np.array(multioutput)
        result = np.average(np.mean(errors, axis=0), weights=multioutput)
    else:  # 'uniform_average' or any other way defaults to a simple average
        result = np.mean(errors)

    return result


from sklearn.metrics import mean_absolute_error as sklearn_mean_absolute_error

# Test data
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# Using our implementation
our_mae = mean_absolute_error(y_true, y_pred)

# Using sklearn's implementation
sklearn_mae = sklearn_mean_absolute_error(y_true, y_pred)

print("Our MAE:", our_mae)
print("Sklearn MAE:", sklearn_mae)

# Test multioutput
y_true = [[0.5, 1], [-1, 1], [7, -6]]
y_pred = [[0, 2], [-1, 2], [8, -5]]

# Using our implementation for multioutput
our_mae_multi = mean_absolute_error(y_true, y_pred)
our_mae_multi_raw = mean_absolute_error(y_true, y_pred)
our_mae_multi_weighted = mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])

# Using sklearn's implementation for multioutput
sklearn_mae_multi = sklearn_mean_absolute_error(y_true, y_pred)
sklearn_mae_multi_raw = sklearn_mean_absolute_error(y_true, y_pred)
sklearn_mae_multi_weighted = sklearn_mean_absolute_error(y_true, y_pred, multioutput=[0.3, 0.7])

print("Our Multioutput MAE:", our_mae_multi)
print("Sklearn Multioutput MAE:", sklearn_mae_multi)
