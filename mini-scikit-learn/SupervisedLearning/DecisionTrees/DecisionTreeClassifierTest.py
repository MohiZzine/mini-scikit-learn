
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

def test():
    # Load the Iris dataset
    data = load_iris()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train our Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)
    print(f"Custom Decision Tree Classifier Accuracy: {accuracy:.4f}")

    # Compare with Scikit-Learn's Decision Tree Classifier
    sklearn_dt = SklearnDecisionTreeClassifier(max_depth=3)
    sklearn_dt.fit(X_train, y_train)
    y_pred_sklearn = sklearn_dt.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Scikit-Learn Decision Tree Classifier Accuracy: {accuracy_sklearn:.4f}")

test()
from sklearn.datasets import load_wine

def test2():
    # Load the Wine dataset
    data = load_wine()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train our Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)
    print(f"Custom Decision Tree Classifier Accuracy on Wine dataset: {accuracy:.4f}")

    # Compare with Scikit-Learn's Decision Tree Classifier
    sklearn_dt = SklearnDecisionTreeClassifier(max_depth=3)
    sklearn_dt.fit(X_train, y_train)
    y_pred_sklearn = sklearn_dt.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Scikit-Learn Decision Tree Classifier Accuracy on Wine dataset: {accuracy_sklearn:.4f}")
    
test2()

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def test3():
        
    # Load the Breast Cancer Wisconsin dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train our Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = dt.predict(X_test)
    accuracy = dt.score(X_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Custom Decision Tree Classifier Accuracy on Breast Cancer dataset: {accuracy:.4f}")
    print(f"Custom Decision Tree Classifier Precision: {precision:.4f}")
    print(f"Custom Decision Tree Classifier Recall: {recall:.4f}")
    print(f"Custom Decision Tree Classifier F1 Score: {f1:.4f}")

    # Compare with Scikit-Learn's Decision Tree Classifier
    sklearn_dt = SklearnDecisionTreeClassifier(max_depth=5)
    sklearn_dt.fit(X_train, y_train)
    y_pred_sklearn = sklearn_dt.predict(X_test)
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    precision_sklearn = precision_score(y_test, y_pred_sklearn)
    recall_sklearn = recall_score(y_test, y_pred_sklearn)
    f1_sklearn = f1_score(y_test, y_pred_sklearn)
    print(f"Scikit-Learn Decision Tree Classifier Accuracy on Breast Cancer dataset: {accuracy_sklearn:.4f}")
    print(f"Scikit-Learn Decision Tree Classifier Precision: {precision_sklearn:.4f}")
    print(f"Scikit-Learn Decision Tree Classifier Recall: {recall_sklearn:.4f}")
    print(f"Scikit-Learn Decision Tree Classifier F1 Score: {f1_sklearn:.4f}")

test3()