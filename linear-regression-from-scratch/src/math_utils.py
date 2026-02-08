import numpy as np

# Calculates the mean squared error loss
def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    

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
