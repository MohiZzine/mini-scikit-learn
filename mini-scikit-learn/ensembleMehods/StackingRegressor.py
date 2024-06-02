import numpy as np
from sklearn.model_selection import cross_val_predict, KFold

class StackingRegressor:
    def __init__(self, estimators, final_estimator=None, cv=None, n_jobs=None, passthrough=False, verbose=0):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.passthrough = passthrough
        self.verbose = verbose

    def fit(self, X, y):
        self.estimators_ = []
        self.named_estimators_ = {}
        
        for name, estimator in self.estimators:
            if estimator == 'drop':
                continue
            estimator.fit(X, y)
            self.estimators_.append(estimator)
            self.named_estimators_[name] = estimator

        # Create a stack of predictions from the base estimators
        if self.cv is None:
            self.cv = 5
        cv = KFold(n_splits=self.cv) if isinstance(self.cv, int) else self.cv
        
        meta_features = np.column_stack([
            cross_val_predict(estimator, X, y, cv=cv, method='predict', n_jobs=self.n_jobs)
            for estimator in self.estimators_
        ])
        
        if self.passthrough:
            meta_features = np.hstack((X, meta_features))

        self.final_estimator_ = self.final_estimator
        self.final_estimator_.fit(meta_features, y)
        
        return self

    def predict(self, X):
        meta_features = np.column_stack([estimator.predict(X) for estimator in self.estimators_])
        
        if self.passthrough:
            meta_features = np.hstack((X, meta_features))
        
        return self.final_estimator_.predict(meta_features)

    def get_params(self, deep=True):
        return {
            'estimators': self.estimators,
            'final_estimator': self.final_estimator,
            'cv': self.cv,
            'n_jobs': self.n_jobs,
            'passthrough': self.passthrough,
            'verbose': self.verbose
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
