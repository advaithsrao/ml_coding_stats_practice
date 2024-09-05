import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Union

from ml import time_ml_training, logger
from ml.regression import SupervisedRegressor
from ml.gradient_descent import BatchGradientDescent
from ml.helpers import (
    convert_to_np_array,
    load_sample_dataset,
    get_regression_results,
)


class LinearRegressor(SupervisedRegressor):
    def __init__(self, learning_rate: float=2e-05):
        super().__init__()
        self.learning_rate = learning_rate
        self.X_train = self.y_train = None
        self.param = None
        self.logger = logger
        
    @time_ml_training
    def fit(
        self, 
        X: Union[List, pd.DataFrame, np.ndarray],
        y: Union[List, pd.DataFrame, np.ndarray],
        num_epochs: int = 3,
    ):
        self.X_train = convert_to_np_array(X)
        self.y_train = convert_to_np_array(y)
        
        # Add a column of 1s for the bias term
        self.X_train = np.insert(self.X_train, 0, 1, axis=1)

        num_samples, num_features = np.shape(self.X_train)
        self.logger.info(f"Number of Samples: {num_samples}, Number of Features: {num_features}")

        self.param = np.random.normal(loc=0, scale=1, size=[num_features, 1])
        
        if len(self.X_train) != len(self.y_train):
            self.logger.info(f"Length Mismatch for X and y with {len(self.X_train)} and {len(self.y_train)}")
        
        gd = BatchGradientDescent(
            param=self.param,
            X=self.X_train,
            y_true=self.y_train,
            y_pred=np.dot(self.X_train, self.param),
            num_epochs=num_epochs,
            learning_rate=self.learning_rate,
            loss="rmse"
        )

        gd.logger = self.logger

        # Run the gradient descent algorithm
        gd.update_param()

        # Update the param with the final values
        self.param = gd.param

    def predict(
        self,
        X: Union[List, pd.DataFrame, np.ndarray]
    ) -> np.array:
        X_test = convert_to_np_array(X)
        return np.dot(
            np.insert(X_test, 0, 1, axis=1),
            self.param
        )

# Example Usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_sample_dataset('regression')
    lr = LinearRegressor(learning_rate=0.01)
    logger = lr.logger
    lr.fit(X_train, y_train, num_epochs=2)
    y_pred = lr.predict(X_test)
    logger.info(f"Regression Results: {get_regression_results(y_test, y_pred)}")