from typing import Union
import numpy as np

from ml import GradientDescent, time_operation, logger

RMSE = "rmse"
BCE = "cross_entropy"

def loss_and_gradients(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, loss: str) -> dict:
    num_samples, _ = X.shape
    loss_gradients_dict = {
        "loss": None,
        "gradients": None
    }
    if loss == RMSE:
        mse = np.mean((y_true - y_pred) ** 2)
        loss_gradients_dict["loss"] = np.sqrt(mse)  # for logging
        loss_gradients_dict["gradients"] = -2 / num_samples * np.dot(X.T, (y_true - y_pred))
    elif loss == BCE:
        eps = 1e-9
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss_gradients_dict["loss"] = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        loss_gradients_dict["gradients"] = 1 / X.shape[0] * np.dot(X.T, (y_pred - y_true))
    else:
        raise ValueError("Invalid Loss Function")
    
    return loss_gradients_dict

class BatchGradientDescent(GradientDescent):
    def __init__(
        self,
        param: Union[float, np.ndarray],
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_epochs: int = 5,
        learning_rate: float = 0.01,
        loss: str = RMSE
    ):
        super().__init__()
        self.param = param
        self.X = X
        self.y_true = y_true
        self.y_pred = y_pred

        if len(self.X) != len(self.y_true):
            raise ValueError(f"Length Mismatch for X and y with {len(self.X)} and {len(self.y_true)}")
        
        if len(self.y_true) != len(self.y_pred):
            raise ValueError(f"Length Mismatch for y_true and y_pred with {len(self.y_true)} and {len(self.y_pred)}")
        
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.loss = loss
        self.logger = logger
        self.iteration_number = 1
    
    @time_operation
    def update_param(self) -> None:
        self.logger.info(f"Running Batch Gradient Descent for {self.num_epochs} epochs")

        # early stopping criteria
        # if the previous iteration's param values are the same as the current iteration's param values to 3 decimal values
        # _tmp_param = None

        while self.iteration_number <= self.num_epochs:
            # _tmp_param = np.round(np.array(self.param))

            self.logger.info(f"Epoch: {self.iteration_number}")

            if self.loss == BCE:
                logits = np.dot(self.X, self.param)
                self.y_pred = 1 / (1 + np.exp(-logits))
            elif self.loss == RMSE:
                self.y_pred = np.dot(self.X, self.param)
            
            _loss_gradients_dict = loss_and_gradients(self.X, self.y_true, self.y_pred, self.loss)
            
            self.logger.info(f"Loss: {_loss_gradients_dict['loss']}")
            self.logger.info(f"Gradients: {_loss_gradients_dict['gradients']}")
            self.param -= self.learning_rate * _loss_gradients_dict["gradients"]

            self.iteration_number += 1
        
        self.logger.info(f"Done running Batch Gradient Descent for {self.num_epochs} epochs")
        self.logger.info(f"Final Loss: {_loss_gradients_dict['loss']}")
        self.logger.info(f"Total Iterations: {self.iteration_number}")
