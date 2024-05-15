from mean_squared_log_error import mean_squared_log_error
import numpy as np

def root_mean_squared_log_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average'):
    """
    Compute root mean squared logarithmic error regression loss.
    """
    msle = mean_squared_log_error(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)
    rmsle = np.sqrt(msle)
    return rmsle


from sklearn.metrics import mean_squared_log_error as sk_msle

# Example usage and testing
y_true = [3, 5, 2.5, 7]
y_pred = [2.5, 5, 4, 8]

rmsle = root_mean_squared_log_error(y_true, y_pred)
print("Root Mean Squared Logarithmic Error (RMSLE):", rmsle)

# Compare with scikit-learn's implementation
sk_rmsle = np.sqrt(sk_msle(y_true, y_pred))
print("Root Mean Squared Logarithmic Error (scikit-learn):", sk_rmsle)
