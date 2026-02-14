from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd

from src.metrics.classification import confusion_matrix, accuracy_score, macro_recall, macro_precision, macro_f1
from src.models.baseline import MajorityClassClassifier
from src.models.knn import KNNClassifier
from src.preprocessing.train_test_split import train_test_split
from src.preprocessing.standard_scaler import StandardScaler

# Load data
X, y = load_iris(return_X_y=True)
n_classes = len(set(y))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit classifiers
maj_clf = MajorityClassClassifier().fit(X_train_scaled, y_train)
knn_clf = KNNClassifier().fit(X_train_scaled, y_train)

# make predictions
y_pred_maj = maj_clf.predict(X_test_scaled)
y_pred_knn = knn_clf.predict(X_test_scaled)



## Printing results ##

print(" ##########################")
print(" #   Comparison Metrics   #")
print(" ##########################")
print()
print(" -- Baseline Confusion Matrix --")
print(confusion_matrix(y_test, y_pred_maj, n_classes))
print()
print(" -- KNN Confusion Matrix -- ")
print(confusion_matrix(y_test, y_pred_knn, n_classes))


