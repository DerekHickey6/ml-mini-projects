from matplotlib import axes, pyplot as plt
from model import LinearRegressionGD
import numpy as np
from data.datasets import load_clean_LR_w_noise

# Load and fit data
X, y = load_clean_LR_w_noise()

reg = LinearRegressionGD(lr=1e-5, epochs=500)
reg.fit(X, y)

# Animation for training model and loss-vs-epoch
for i in range(len(reg.param_history_)):
    y_pred = reg.param_history_[i][0] * X + reg.param_history_[i][1]
    plt.clf()

    plt.title(f"Epoch: {i}/{reg.epochs} - Loss: {reg.loss_history_[i]:0.2f} - lr: {reg.lr}")
    plt.xlabel("X data")

    # Loss curve vs epocjs
    plt.plot(X[0:i], 8*np.log(reg.loss_history_[0:i]), color='green', label='Scaled Log(loss)')
    # Training of model
    plt.plot(X, y_pred, color='red', label='Model Learning')
    # True data
    plt.scatter(X, y, s = 0.5, label='True Data')
    plt.legend()
    plt.pause(0.05)

plt.show()
