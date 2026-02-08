from sklearn.model_selection import train_test_split
from data.datasets import load_clean_LR_w_noise
from math_utils import mse, rmse, mae, r2
import pandas as pd
from model import LinearRegressionGD

# Load data
X_data, y_data = load_clean_LR_w_noise()

# Split to training
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

# Fit data
reg = LinearRegressionGD(lr=3e-5, epochs=300)
reg.fit(X_train, y_train)

# Make predictions
train_pred = reg.predict(X_train)
test_pred = reg.predict(X_test)

# Calculate metrics
train_mse = mse(y_true=y_train, y_pred=train_pred)
test_mse = mse(y_true=y_test, y_pred=test_pred)

train_rmse = rmse(y_true=y_train, y_pred=train_pred)
test_rmse = rmse(y_true=y_test, y_pred=test_pred)

train_mae = mae(y_true=y_train, y_pred=train_pred)
test_mae = mae(y_true=y_test, y_pred=test_pred)

train_r2 = r2(y_true=y_train, y_pred=train_pred)
test_r2 = r2(y_true=y_train, y_pred=train_pred)

# Display Evaluation metrics
eval = pd.DataFrame([[train_mse, test_mse], [train_rmse, test_rmse], [train_mae, test_mae], [train_r2, test_r2]],
                    index=['MSE', 'RMSE','MAE','R^2'],
                    columns=["Train", "Test"])
eval["Difference"] = eval["Train"] - eval["Test"]
print("     --- Evaluation Metrics --- ")
print(eval)
