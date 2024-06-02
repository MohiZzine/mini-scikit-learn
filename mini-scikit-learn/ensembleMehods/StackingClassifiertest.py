from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier as SklearnStackingClassifier
from sklearn.tree import DecisionTreeClassifier
from StackingClassifier import StackingClassifier   
# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base classifiers
base_classifiers = [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=3)]

# Train and test 
your_model = StackingClassifier(base_classifiers=base_classifiers)
your_model.fit(X_train, y_train)
your_predictions = your_model.predict(X_test)
your_accuracy = accuracy_score(y_test, your_predictions)

# Train and test scikit-learn's implementation
sklearn_model = SklearnStackingClassifier(estimators=[('dt1', DecisionTreeClassifier(max_depth=1)), ('dt3', DecisionTreeClassifier(max_depth=3))],
                                           final_estimator=DecisionTreeClassifier(max_depth=1))
sklearn_model.fit(X_train, y_train)
sklearn_predictions = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)

print("StackingClassifier Accuracy:", your_accuracy)
print("Scikit-learn StackingClassifier Accuracy:", sklearn_accuracy)