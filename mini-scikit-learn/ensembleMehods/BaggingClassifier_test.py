from BaggingClassifier import BaggingClassifier
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from SupervisedLearning.DecisionTrees.DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np

# Example data
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, size=100)

# Create and train BaggingClassifier
bagging_clf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42, oob_score=True)
bagging_clf.fit(X, y)

# Make predictions
predictions = bagging_clf.predict(X)
proba_predictions = bagging_clf.predict_proba(X)
score = bagging_clf.score(X, y)

# Print results for manual inspection
print("Predictions:\n", predictions)
print("Predicted Probabilities:\n", proba_predictions)
print("Score:\n", score)
print("OOB Score:\n", bagging_clf.oob_score_)
