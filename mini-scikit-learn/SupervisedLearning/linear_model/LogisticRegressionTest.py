
# Testing the implementation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression
def test():
    # Load the Iris dataset
    data = load_iris()
    X, y = data.data, (data.target == 2).astype(int)  # Binary classification (class 2 vs. others)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Initialize and train our Logistic Regression
    lr = LogisticRegression(learning_rate=0.01, max_iter=1000)
    lr.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = lr.predict(X_test)
    accuracy = lr.score(X_test, y_test)
    print(f"Custom Logistic Regression Accuracy: {accuracy:.4f}")

    # Compare with Scikit-Learn's Logistic Regression
    sklearn_lr = SklearnLogisticRegression()
    sklearn_lr.fit(X_train, y_train)
    y_pred_sklearn = sklearn_lr.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Scikit-Learn Logistic Regression Accuracy: {accuracy_sklearn:.4f}")


test()
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import  precision_score, recall_score, f1_score

def test2():
        
    # Load the Breast Cancer Wisconsin dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Initialize and train our Logistic Regression
    lr = LogisticRegression(learning_rate=0.01, max_iter=1000)
    lr.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = lr.predict(X_test)
    accuracy = lr.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Custom Logistic Regression Accuracy on Breast Cancer dataset: {accuracy:.4f}")
    print(f"Custom Logistic Regression Precision: {precision:.4f}")
    print(f"Custom Logistic Regression Recall: {recall:.4f}")
    print(f"Custom Logistic Regression F1 Score: {f1:.4f}")

    # Compare with Scikit-Learn's Logistic Regression
    sklearn_lr = SklearnLogisticRegression(max_iter=1000)
    sklearn_lr.fit(X_train, y_train)
    y_pred_sklearn = sklearn_lr.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    precision_sklearn = precision_score(y_test, y_pred_sklearn)
    recall_sklearn = recall_score(y_test, y_pred_sklearn)
    f1_sklearn = f1_score(y_test, y_pred_sklearn)
    print(f"Scikit-Learn Logistic Regression Accuracy on Breast Cancer dataset: {accuracy_sklearn:.4f}")
    print(f"Scikit-Learn Logistic Regression Precision: {precision_sklearn:.4f}")
    print(f"Scikit-Learn Logistic Regression Recall: {recall_sklearn:.4f}")
    print(f"Scikit-Learn Logistic Regression F1 Score: {f1_sklearn:.4f}")   
    
    
test2()