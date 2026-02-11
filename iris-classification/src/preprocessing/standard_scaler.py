import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = np.sqrt(self.var_) if self.var_ != 0 else 1
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

        self.mean_ = np.mean(X)
        self.var_ = np.var(X)
        self.n_features_ = X.shape[1]

        return self

    # returns X_scaled
    def transform(self, X):
        # Enforces fit call before transform
        if self.mean_ is None or self.var_ is None or self.n_features_ is None:
            raise ValueError("Fit() must be called before transform()")

        # Ensures data to be transformed has same num features
        if X.shape[1] is not self.n_features_:
            raise ValueError("Number of features does not match number learned in fit()")

        return (X - self.mean_) / self.std_

    # Convenience method for first time fitting + transforming
    def fit_transform(self, X):
        self.mean_ = np.mean(X)
        self.var_ = np.var(X)



        return (X - self.mean_) / self.std_