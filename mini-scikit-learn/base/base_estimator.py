class BaseEstimator:
    """
    Base class for all estimators.
    """
    def fit(self, data):
        raise NotImplementedError("fit method not implemented")
    
    def get_params(self):
        return self.__dict__
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
