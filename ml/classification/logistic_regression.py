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
    get_classification_results,
)

class LogisticRegressor(SupervisedRegressor):
    def __init__(self, learning_rate: float = 0.01):
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

        # Add bias term
        self.X_train = np.insert(self.X_train, 0, 1, axis=1)

        num_samples, num_features = np.shape(self.X_train)
        self.logger.info(f"Number of Samples: {num_samples}, Number of Features: {num_features}")

        self.param = np.random.normal(loc=0, scale=1, size=[num_features, 1])

        if len(self.X_train) != len(self.y_train):
            self.logger.warning(f"Length Mismatch for X and y with {len(self.X_train)} and {len(self.y_train)}")

        # Initial prediction using sigmoid
        logits = np.dot(self.X_train, self.param)
        y_pred = 1 / (1 + np.exp(-logits))
        self.logger.info(f"Shapes of Outputs: {np.shape(y_pred) = } and {np.shape(y) = }")

        gd = BatchGradientDescent(
            param=self.param,
            X=self.X_train,
            y_true=self.y_train,
            y_pred=y_pred,
            num_epochs=num_epochs,
            learning_rate=self.learning_rate,
            loss="cross_entropy"
        )

        gd.logger = self.logger

        gd.update_param()

        self.param = gd.param

    def predict(
        self,
        X: Union[List, pd.DataFrame, np.ndarray],
        threshold: float = 0.5
    ) -> np.ndarray:
        X_test = convert_to_np_array(X)
        X_test = np.insert(X_test, 0, 1, axis=1)

        logits = np.dot(X_test, self.param)
        probabilities = 1 / (1 + np.exp(-logits))

        # Return binary predictions
        return (probabilities >= threshold).astype(int)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_sample_dataset('classification')
    log_reg = LogisticRegressor(learning_rate=0.01)
    logger = log_reg.logger
    log_reg.fit(X_train, y_train, num_epochs=50)
    y_pred = log_reg.predict(X_test)
    logger.info(f"Classification Accuracy: {get_classification_results(y_test, y_pred)}")
