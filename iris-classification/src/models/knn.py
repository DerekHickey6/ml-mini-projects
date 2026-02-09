# KNN Classifier from scratch
import numpy as np

class KNNClassifier:
    def __init__(self, k_val=5):
        self.k_val = k_val
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Validate X and y
        if not (len(X) and len(y)):
            raise ValueError("Features and labels can not be empty")

        if not (len(X) == len(y)):
            raise ValueError("Shape mismatch: Features and labels must be length")

        if self.k_val > len(X):
            raise ValueError("K Value can not be greater than number of objects")

        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        return self

    def predict(self, X):

        distances = [np.linalg.norm(X - Xts) for Xts in self.X_train]
        

