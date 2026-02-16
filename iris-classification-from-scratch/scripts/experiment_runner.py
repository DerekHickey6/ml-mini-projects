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

maj_conf_mat = confusion_matrix(y_test, y_pred_maj, n_classes)
knn_conf_mat = confusion_matrix(y_test, y_pred_knn, n_classes)

# Create model + metrics dictionarys for modular metric processing and display
model_preds = {"baseline": y_pred_maj, "knn": y_pred_knn}
metrics_df = pd.DataFrame()

# Calculate metrics
for i in model_preds:
    metrics = {}

    conf_mat = confusion_matrix(y_test, model_preds[i], n_classes)
    metrics['Accuracy'] = accuracy_score(y_test, model_preds[i])
    metrics['macro precision'] = macro_precision(conf_mat)
    metrics['macro recall'] = macro_recall(conf_mat)
    metrics['macro f1'] = macro_f1(conf_mat)

    metrics_df[i] = pd.Series(metrics)




## Printing results ##

print(" ##########################")
print(" #   Comparison Metrics   #")
print(" ##########################")
print()
print(" -- Baseline Confusion Matrix --")
print(maj_conf_mat)
print()
print(" -- KNN Confusion Matrix -- ")
print(knn_conf_mat)
print()
print(metrics_df)

