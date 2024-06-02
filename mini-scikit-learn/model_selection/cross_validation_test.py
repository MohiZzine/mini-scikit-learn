from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score as sklearn_cross_val_score, KFold as SklearnKFold
from cross_validation import cross_val_score, KFold
# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Initialize the custom k-fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform custom cross-validation
custom_scores = cross_val_score(model, X, y, cv=kf)
print(f"Custom cross-validation scores: {custom_scores}")
print(f"Mean custom cross-validation score: {custom_scores.mean():.4f}")

# Perform cross-validation with Scikit-Learn's implementation
sklearn_kf = SklearnKFold(n_splits=5, shuffle=True, random_state=42)
sklearn_scores = sklearn_cross_val_score(model, X, y, cv=sklearn_kf)
print(f"Scikit-Learn cross-validation scores: {sklearn_scores}")
print(f"Mean Scikit-Learn cross-validation score: {sklearn_scores.mean():.4f}")
