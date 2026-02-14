import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=False, random_seed=1):
    X = np.array(X)
    y = np.array(y)
    X_train = []
    X_test = []
    y_train = []
    y_test = []


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

    classes = np.unique(y)

    # Separates data on index of label and splits on test_split, concatenate on appropriate list
    for i in classes:
        class_idx = np.where(y == i)
        class_split_index = int(len(class_idx[0]) * (1 - test_size))

        # Create list of split indivuidual class arrays
        X_train.append(X[class_idx][:class_split_index])
        y_train.append(y[class_idx][:class_split_index])
        X_test.append(X[class_idx][class_split_index:])
        y_test.append(y[class_idx][class_split_index:])

    # Concatenate all arrays into splits
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    # Shuffle Data
    if shuffle:
        rng = np.random.default_rng(random_seed)

        # Create shuffles indices
        idxs_train = rng.permutation(len(X_train))
        idxs_test = rng.permutation(len(X_test))

        X_train = X_train[idxs_train]
        y_train = y_train[idxs_train]
        X_test = X_test[idxs_test]
        y_test = y_test[idxs_test]


    return X_train, X_test, y_train, y_test






