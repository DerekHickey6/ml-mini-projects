from sklearn.model_selection import train_test_split
from data.datasets import load_clean_LR_w_noise
from math_utils import mse, rmse, mae, r2
import pandas as pd
from model import LinearRegressionGD

# Load data
X_data, y_data = load_clean_LR_w_noise()

# Split to training
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Fit data to training data
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
test_r2 = r2(y_true=y_test, y_pred=test_pred)

# Display Evaluation metrics
eval_df = pd.DataFrame([[test_mse, train_mse], [test_rmse, train_rmse], [test_mae, train_mae], [test_r2, train_r2]],
                    index=['MSE', 'RMSE','MAE','R²'],
                    columns=["Test", "Train"])

eval_df["Difference"] = eval_df["Test"] - eval_df["Train"]

print("     --- Evaluation Metrics --- ")
print(eval_df)
print(f"Final Weight: {reg.w_:0.4f}")
print(f"Final Bias:   {reg.b_:0.4f}")

