import numpy as np

def confusion_matrix(y_true, y_pred, n_classes):
    # Checking Shape error
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} and {y_pred.shape} do not match")

    # Checking bounds
    if y_true.min() < 0 or y_true.max() >= n_classes:
        raise ValueError("y_true contains labels outside valid range [0, n_classes)")

    if y_pred.min() < 0 or y_pred.max() >= n_classes:
        raise ValueError("y_pred contains labels outside valid range [0, n_classes)")

    # Initialize matrix
    conf_mat = np.zeros((n_classes, n_classes), dtype=int)

    # Count true vs predicted
    for i, j in zip(y_true, y_pred):
        conf_mat[i, j] += 1

    return conf_mat

def accuracy_score(y_true, y_pred):
    pass


