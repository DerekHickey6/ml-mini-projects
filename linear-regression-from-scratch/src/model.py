from re import X
from tkinter import W
from math_utils import mse, compute_gradients
import numpy as np
from sklearn.model_selection import train_test_split

X_data = [2, 5.6, 7.8, 9.9, 10, 11]
y_data = [3, 4.16, 7.28, 11.9, 12, 13.8]
X_test = [x+2 for x in X_data]


class LinearRegressionGD:
    def __init__(self, lr=0.005, epochs=5000):
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
            # if epoch % 50 == 0:
            #     print(f"Weight: {self.w_}, Bias: {self.b_}")
            #     print(f"Loss: {loss}")


            # Store parameters as tuple ever other iteration
            if epoch % 2 == 0:
                self.param_history_.append((self.w_, self.b_))

        return self

    def predict(self, X):
        if not (self.b_ and self.w_):
            raise Exception("An error occured: Model not trained")

        X = np.asarray(X)
        y_preds = self.w_ * X + self.b_
        return y_preds


reg = LinearRegressionGD().fit(X_data, y_data)
print(reg.predict(X_test))

