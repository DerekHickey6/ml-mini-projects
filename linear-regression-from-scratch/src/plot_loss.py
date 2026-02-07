from matplotlib import axes, pyplot as plt
from model import LinearRegressionGD
import numpy as np
from data.datasets import load_clean_LR_w_noise
from sklearn.model_selection import train_test_split

X_data, y_data = load_clean_LR_w_noise()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

reg = LinearRegressionGD()
reg.fit(X_train, y_train)

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
axes[0].plot(range(len(reg.loss_history_)), np.log(reg.loss_history_))
axes[0].set_title('Log of loss curve')
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Log(loss)")

axes[1].plot(range(len(reg.param_history_)), [w for w, b in reg.param_history_])
axes[1].set_title('Weights vs Epoch')
axes[1].set_xlabel("Epochs")

axes[2].plot(range(len(reg.param_history_)), [b for w, b in reg.param_history_])
axes[2].set_title('Bias vs Epoch')
axes[2].set_xlabel("Epochs")

# axes[1].plot(, -y)
# axes[1].set_title('Subplot 2')

plt.savefig("loss_weight_bias.png")
plt.tight_layout()
plt.show()
