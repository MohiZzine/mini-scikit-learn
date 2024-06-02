from sklearn.ensemble import VotingClassifier as SklearnVotingClassifier
import numpy as np

class VotingClassifier:
    def __init__(self, estimators, voting='hard'):
        self.estimators = estimators
        self.voting = voting

    def fit(self, X, y):
        for _, estimator in self.estimators:
            estimator.fit(X, y)

    def predict(self, X):
        if self.voting == 'soft':
            predictions = np.argmax(np.mean([estimator.predict_proba(X) for _, estimator in self.estimators], axis=0), axis=1)
        else:  # 'hard' voting
            predictions = np.mean([estimator.predict(X) for _, estimator in self.estimators], axis=0)
            predictions = np.round(predictions).astype(int)
        return predictions
