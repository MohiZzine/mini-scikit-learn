import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.sparse.linalg import LinearOperator, cg  # Conjugate gradient solver

def sigmoid(z):
    # Clip z to avoid overflow in the exponential function when z is very large or very small
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self, penalty='l2', C=1.0, fit_intercept=True, intercept_scaling=1,
                 solver='lbfgs', max_iter=100, multi_class='auto', tol=1e-4, random_state=None, l1_ratio=None):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.tol = tol
        self.random_state = random_state
        self.l1_ratio = l1_ratio
        self.coef_ = None
        self.intercept_ = None

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1)) * self.intercept_scaling
        return np.hstack((intercept, X))

    def _logistic_loss(self, w, X, y):
        m = len(y)
        z = X.dot(w)
        log_likelihood = -np.sum(np.log(1 + np.exp(-y * z)))
        grad = X.T.dot(y * (1 - 1 / (1 + np.exp(-y * z)))) / m

        if self.penalty == 'l2':
            log_likelihood -= 0.5 * self.C * np.sum(w ** 2)
            grad -= self.C * w
        elif self.penalty == 'l1':
            log_likelihood -= self.C * np.sum(np.abs(w))
        elif self.penalty == 'elasticnet':
            l1 = self.C * self.l1_ratio
            l2 = self.C * (1 - self.l1_ratio)
            log_likelihood -= l1 * np.sum(np.abs(w)) + 0.5 * l2 * np.sum(w ** 2)
            grad -= l1 * np.sign(w) + l2 * w

        return log_likelihood, grad

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)) * self.intercept_scaling, X])

        options = {'maxiter': self.max_iter, 'disp': False}
        initial_coef = np.zeros(X.shape[1])

        if self.solver == 'lbfgs':
            result = minimize(lambda w: self._logistic_loss(w, X, y), initial_coef, method='L-BFGS-B', jac=True, options=options)
        elif self.solver == 'newton-cg':
            def Hessian(w):
                Xw = X.dot(w)
                s = 1 / (1 + np.exp(-Xw))
                R = np.diag(s * (1 - s))
                return X.T.dot(R).dot(X)
            
            result = minimize(lambda w: self._logistic_loss(w, X, y), initial_coef, method='Newton-CG',
                              jac=True, hess=Hessian, options=options)
        # Implement more solvers like 'liblinear' and 'saga'

        if self.fit_intercept:
            self.intercept_ = result.x[0]
            self.coef_ = result.x[1:]
        else:
            self.coef_ = result.x

        return self

    def predict_proba(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        z = np.dot(X, np.hstack([self.intercept_, self.coef_]))
        proba = 1 / (1 + np.exp(-z))
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    



import numpy as np
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Assuming CustomLogisticRegression is defined as per the previous implementation details
# from your_module import CustomLogisticRegression

def test_logistic_regression(custom_model, sklearn_model, X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the custom model
    custom_model.fit(X_train, y_train)
    custom_predictions = custom_model.predict(X_test)
    custom_accuracy = accuracy_score(y_test, custom_predictions)

    # Fit the sklearn model
    sklearn_model.fit(X_train, y_train)
    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

    print("Custom Model Accuracy:", custom_accuracy)
    print("Sklearn Model Accuracy:", sklearn_accuracy)
    print("---" * 10)

# Test case 1: Binary classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
custom_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs')
sklearn_model = SklearnLogisticRegression(penalty='l2', C=1.0, solver='lbfgs')
print("Binary Classification Test:")
test_logistic_regression(custom_model, sklearn_model, X, y)

# Test case 2: Multiclass classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=4, random_state=42)
custom_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', multi_class='ovr')
sklearn_model = SklearnLogisticRegression(penalty='l2', C=1.0, solver='lbfgs', multi_class='ovr')
print("Multiclass Classification Test:")
test_logistic_regression(custom_model, sklearn_model, X, y)

# Test case 3: Regularization effects with L1 penalty
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, random_state=42)
custom_model = LogisticRegression(penalty='l1', C=0.5, solver='saga')  # Assuming 'saga' is implemented
sklearn_model = SklearnLogisticRegression(penalty='l1', C=0.5, solver='saga')
print("L1 Regularization Test:")
test_logistic_regression(custom_model, sklearn_model, X, y)


