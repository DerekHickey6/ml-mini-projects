from data.datasets import load_clean_LR_w_noise
from math_utils import mse, compute_gradients
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_data, y_data = load_clean_LR_w_noise()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

class LinearRegressionGD:
    def __init__(self, lr=5e-5, epochs=1000):
        self.w_ = None
        self.b_ = None
        self.lr = lr
        self.epochs = epochs

        # Track loss
        self.loss_history_ = []
        self.param_history_ = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # Initialize weight and bias
        self.w_ = 0
        self.b_ = 0


        # Clear loss in case fit is called twice
        self.loss_history_ = []

        for epoch in range(self.epochs):
            # Make Predictions
            y_preds = self.w_ * X + self.b_

            # Compute the loss
            loss = mse(y_true=y, y_pred=y_preds)

            # Add loss to history
            self.loss_history_.append(loss)

            # Adjust weight and bias based on lr * grads
            dw_db_grads = compute_gradients(X, y, self.w_, self.b_)
            self.w_ += dw_db_grads[0] * self.lr * -1
            self.b_ += dw_db_grads[1] * self.lr * -1

            ## Testing ##
            if epoch % 10 == 0:
                print(f"Weight: {self.w_}, Bias: {self.b_}")
                print(f"Loss: {loss}")


            # Store parameters as tuple ever other iteration
            self.param_history_.append((self.w_, self.b_))

        return self

    def predict(self, X):
        if not (self.b_ and self.w_):
            raise Exception("An error occured: Model not trained")

        X = np.asarray(X)
        y_preds = self.w_ * X + self.b_
        return y_preds

## Test ##
# reg = LinearRegressionGD().fit(X_train, y_train)
# print(np.sort(reg.predict(X_test)))

