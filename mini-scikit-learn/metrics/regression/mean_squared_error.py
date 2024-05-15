import numpy as np

def mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Compute mean squared error regression loss.
    
    Args:
    y_true (array-like): Ground truth (correct) target values.
    y_pred (array-like): Estimated target values.
    sample_weight (array-like, optional): Sample weights.
    multioutput (str or array-like): Defines aggregating of multiple output values.
    
    Returns:
    float or ndarray: Mean squared error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate errors
    errors = (y_true - y_pred) ** 2
    
    # Apply sample weights
    if sample_weight is not None:
        sample_weight = np.array(sample_weight)
        assert sample_weight.shape[0] == errors.shape[0], "sample_weight and errors must have the same length"
        errors = errors * sample_weight

    # Compute mean squared error
    if multioutput == 'raw_values':
        mse = np.mean(errors, axis=0)
    elif isinstance(multioutput, (list, np.ndarray)):
        weights = np.array(multioutput)
        assert weights.shape[0] == errors.shape[1], "weights and errors must have the same number of columns"
        mse = np.average(np.mean(errors, axis=0), weights=weights)
    else: # uniform_average
        mse = np.mean(errors)

    return mse
