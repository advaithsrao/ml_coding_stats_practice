import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Union

from ml import time_ml_training, logger
from ml.classification import SupervisedClassifier
from ml.helpers import (
    convert_to_np_array, 
    euclidean_distance, 
    load_sample_dataset, 
    get_classification_results
)


class KNN(SupervisedClassifier):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.X_train = self.y_train = None
        self.logger = logger
        
    @time_ml_training
    def fit(
        self, 
        X: Union[List, pd.DataFrame, np.ndarray],
        y: Union[List, pd.DataFrame, np.ndarray]
    ):
        self.X_train = convert_to_np_array(X)
        self.y_train = convert_to_np_array(y)
        if len(self.X_train) != len(self.y_train):
            self.logger.info(f"Length Mismatch for X and y with {len(self.X_train)} and {len(self.y_train)}")

    def predict(
        self,
        X: Union[List, pd.DataFrame, np.ndarray]
    ) -> np.array:
        X_test = convert_to_np_array(X)
        preds = []
        
        for _x in X_test:
            preds.append(self._predict(_x))
        
        return np.array(preds)

    def _predict(
        self, 
        x: np.ndarray
    ):
        distances = np.argsort(
            np.array(
                [euclidean_distance(x, _X) for _X in self.X_train]
            )
        )[:self.k]

        return Counter(self.y_train[distances]).most_common(1)[0][0]


# Example Usage
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_sample_dataset('classification')
    knn = KNN(k=17)
    logger = knn.logger
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    logger.info(f"Classification Results: {get_classification_results(y_test, y_pred)}")