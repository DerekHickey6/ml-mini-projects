import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from metrics.classification import confusion_matrix
from models.knn import KNNClassifier
from preprocessing.train_test_split import train_test_split

X, y = load_iris(return_X_y=True)
n_classes = len(set(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# clf = KNNClassifier().fit(X_train_scaled, y_train)
# y_pred = clf.predict(X_test_scaled)

print(X)
print(y)



