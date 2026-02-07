
import numpy as np

def load_clean_LR_w_noise(
    return_X_y=True,
    n_samples=50,
    random_state=None,
    noise_std=1.5,
    weight = 1.5,
    bias = 2
    ):

    if random_state is not None:
        np.random.seed(random_state)

    X_data = np.arange(n_samples, step=0.1)
    noise = np.asarray(np.random.random(len(X_data)) * noise_std)
    y_data = (weight * X_data + bias) + noise

    if return_X_y:
        return X_data, y_data
    else:
        return {
            "X": X_data,
            "y": y_data,
            "true_w": 1.5,
            "true_b": 2.0,
            "noise_std": noise_std
        }