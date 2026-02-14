from src.metrics.classification import confusion_matrix
from src.models.knn import KNNClassifier
import numpy as np

## Small Dataset Test ##
X_train = np.array([[1, 1], [1, 2], [2, 1], [2, 2],
                    [8, 8], [8, 9], [9, 8], [9, 9]])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

X_test = np.array([[1.5, 1.5], [8.5, 8.5]])
y_test = np.array([0, 1])

clf = KNNClassifier().fit(X_train, y_train)
y_preds = clf.predict(X_test)

print(" -- small database test --")
print(f"True: {y_test}")
print(f"Pred: {y_preds}")
print()

## Tie-break Test ##
X_train = np.array([[1, 2], [1, 3],
                    [2, 1], [3, 1]])
y_train = np.array([0, 0, 1, 1])

X_test = np.array([[1.5, 1], [1, 1.5]])
y_test = np.array([1, 0])

clf = KNNClassifier(k_val=2).fit(X_train, y_train)
y_preds = clf.predict(X_test)

print(" -- Tie-break test --")
print(f"True: {y_test}")
print(f"Pred: {y_preds}")

