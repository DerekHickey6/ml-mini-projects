import numpy as np

# Used to scale data for appropriate training in KNNClassifier
class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.var_ = None
        self.n_features_ = None

    # Sets mean and var
    def fit(self, X):
        # ensures X is 2D array
        if X.ndim != 2:
            raise ValueError("Shape mismatch: X must be shape (N, D)")

        # Enforces array shape
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("Shape mismatch: Array must have at least 1 row and 1 column")

        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X,axis=0)
        self.n_features_ = X.shape[1]
        self.std_ = np.sqrt(self.var_)

        # Handles zero-variance cases to avoid div by 0
        self.std_[self.std_ == 0] = 1

        return self

    # returns X_scaled
    def transform(self, X):
        # Enforces fit call before transform
        if self.std_ is None:
            raise ValueError("Fit() must be called before transform()")

        # ensures X is 2D array
        if X.ndim != 2:
            raise ValueError("Shape mismatch: X must be shape (N, D)")

        # Ensures data to be transformed has same num features
        if X.shape[1] != self.n_features_:
            raise ValueError("Number of features does not match number learned in fit()")

        # Enforces array shape
        if X.shape[0] < 1 or X.shape[1] < 1:
            raise ValueError("Shape mismatch: Array must have at least 1 row and 1 column")

        return (X - self.mean_) / self.std_

    # Convenience method for first time fitting + transforming
    def fit_transform(self, X):

        self.fit(X)

        return self.transform(X)