import numpy as np

def confusion_matrix(y_true, y_pred, n_classes):
    # Test for Shape error
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} and {y_pred.shape} do not match")

    # Test for bounds
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
    # Test for Shape error
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} and {y_pred.shape} do not match")


    # Test for empty input
    if not (y_true and y_pred):
        raise ValueError("Input must be non-empty")

    # Count the matching pairs and divide by total
    accuracy = np.sum([x == y for x, y in zip(y_true, y_pred)]) / len(y_true)

    return accuracy

def macro_precision(conf_mat):

    accum_precisions = 0
    length = len(conf_mat)

    # Sum of all precisions
    for i in range(length):
        tp = conf_mat[i, i]
        fp = np.sum(conf_mat[:, i]) - tp

        if (tp + fp) != 0:
            accum_precisions += tp / (tp + fp)
        else:
            accum_precisions += 0

    return accum_precisions / length


def macro_recall(conf_mat):

    accum_recall = 0
    length = len(conf_mat)

    # Sum of all recalls
    for i in range(length):
        tp = conf_mat[i, i]
        fn = np.sum(conf_mat[i, :]) - tp

        if (tp + fn) != 0:
            accum_recall += tp / (tp + fn)
        else:
            accum_recall += 0

    return accum_recall / length

def f1(conf_mat):
    pass

