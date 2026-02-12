import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=False, random_seed=1):
    # Calculate split index
    length = len(X)
    split_index = length * test_size

    X = np.array(X)
    y = np.array(y)

    # Shuffle Data
    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(X)
        np.random.shuffle(y)

    # Split the dataa
    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]



    return X_train, X_test, y_train, y_test


