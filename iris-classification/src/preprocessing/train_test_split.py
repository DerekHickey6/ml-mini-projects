import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=False, random_seed=1):
    X = np.array(X)
    y = np.array(y)

    # Ensure test_size is valid
    if test_size > 1 or test_size < 0:
        raise ValueError("Test_size must be between 0 and 1")

    # ensures X is 2D array
    if X.ndim != 2:
        raise ValueError("Shape mismatch: X must be shape (N, D)")

    # Enforces array shape
    if X.shape[0] < 1 or X.shape[1] < 1:
        raise ValueError("Shape mismatch: Array must have at least 1 row and 1 column")

    if y.ndim != 1:
        raise ValueError("y should be 1-dimension")

    if len(X) != len(y):
        raise ValueError("X data and y data must be the same length")

    # Calculate split index
    length = len(X)
    split_index = int(length * (1 - test_size))

    # Shuffle Data
    if shuffle:
        np.random.seed(random_seed)

        idxs = list(range(length))
        np.random.shuffle(idxs)

        X = X[idxs]
        y = y[idxs]

    # Split the data
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


