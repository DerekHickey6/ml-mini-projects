import numpy as np

# Calculates the mean squared error loss
def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred) ** 2)

# Calculate the mean absolute error loss
def mae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(abs(y_true - y_pred))

# Calculate the root mean squared error loss
def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return mse(y_true, y_pred) ** 0.5

# Calculate the R-squared
def r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    var_y = np.var(y_true)

    if var_y == 0:
        return 0
    else:
        return 1 - (mse(y_true, y_pred)/np.var(y_true))

# Compute gradients for weight and bias on given dataset
def compute_gradients(X, y, w, b) -> tuple:
    """
    Returns: (dw, db)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Gradients for descent
    db = 2*np.mean((X * w + b) - y)
    dw = 2*np.mean(X*((X * w + b) - y))

    return (dw, db)
