from math_utils import mse, compute_gradients
import numpy as np

# Linear Regression model wtih Gradient Descent
class LinearRegressionGD:
    def __init__(self, lr=5e-5, epochs=1000, eval_rate=100, eval=True):
        self.w_ = None
        self.b_ = None
        self.lr = lr
        self.eval_rate = eval_rate
        self.eval = eval
        self.epochs = epochs

        # Track loss and parameters for visualization
        self.loss_history_ = []
        self.param_history_ = []

    # Training loop for given data
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

            # Gradient Descent
            dw_db_grads = compute_gradients(X, y, self.w_, self.b_)
            self.w_ += dw_db_grads[0] * self.lr * -1
            self.b_ += dw_db_grads[1] * self.lr * -1

            # Evaluation
            if epoch % self.eval_rate == 0 and eval == True:
                print(f"--- Evaluation at Epoch {epoch} ---")
                print(f"Weight: {self.w_:0.2f} - Bias: {self.b_:0.2f} - Loss: {loss:0.2f}")
                print()

            # Store parameters as tuple ever other iteration
            self.param_history_.append((self.w_, self.b_))

        return self

    # Predict based on given data
    def predict(self, X):
        if not (self.b_ and self.w_):
            raise Exception("An error occured: Model not trained")

        X = np.asarray(X)
        y_preds = self.w_ * X + self.b_
        return y_preds

