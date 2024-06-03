import numpy as np

def mean_squared_log_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Ensure non-negative values since the log of negative numbers is undefined
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("Mean Squared Logarithmic Error cannot be used with negative values.")

    # Apply the log1p transformation
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)

    # Compute squared logarithmic error
    log_errors = (y_true_log - y_pred_log) ** 2

    # Apply weights if provided
    if sample_weight is not None:
        log_errors *= np.array(sample_weight)

    # Aggregate errors
    if multioutput == 'raw_values':
        result = np.mean(log_errors, axis=0)
    else:
        result = np.mean(log_errors)

    # Depending on the 'squared' parameter return MSLE or RMSLE
    if squared:
        return result
    else:
        return np.sqrt(result)


