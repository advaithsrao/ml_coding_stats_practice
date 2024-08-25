import random
import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Union

from ml import time_ml_training, logger
from ml.classification import UnsupervisedClassifier
from ml.helpers import (
    convert_to_np_array, 
    euclidean_distance,
    load_sample_dataset,
)

# #KMeans Algorithm
# 1. Initialize k centroids.
# 2. Define distance criteria, inertia.
# 3. While not convergence,
#     a. Find distance of all points from each of the centroids.
#     b. Assign points to centroids based on min distance to cluster.
#     c. Find new centroids using mean of the clusters.



class KMeans(UnsupervisedClassifier):
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.centroids = {id: None for id in range(self.k)}
        self.clusters = {id: [] for id in range(self.k)}
        self.X_train = self.y_train = None
        self.logger = logger
        self.iteration_number = 1
        
    @time_ml_training
    def fit(
        self, 
        X: Union[List, pd.DataFrame, np.ndarray],
        max_iterations: int = 100,
    ):
        self.X_train = convert_to_np_array(X)
        n_features = np.shape(self.X_train)[1]

        # Initialize centroids with random points from dataset
        ids = random.sample(range(len(self.X_train)), self.k)
        for idx, _id in enumerate(ids):
            self.centroids[idx] = self.X_train[_id]
        
        self.logger.info(f"Initialized Centroids: {self.centroids}")

        # Iterates until either
        #  1. previous interation centroid vals are not the same as the current till 3 decimal points or
        #  2. max_iterations have been reached
        _tmp_centroids = np.zeros([self.k, n_features])
        while not bool(
            np.equal(
                _tmp_centroids, 
                np.round(
                    np.array(list(self.centroids.values())),
                    3
                )
            ).all()
        ) and self.iteration_number <= max_iterations:
            self.logger.info(f"*** Iteration Number ***: {self.iteration_number}")
            _tmp_centroids = np.round(np.array(list(self.centroids.values())), 3)

            # M-step
            for _x in self.X_train:
                closest_cluster = self.get_closest_cluster(_tmp_centroids, _x)
                self.clusters[closest_cluster].append(_x)
            self.logger.info("Clusters Assigned")

            # E-step
            for _cluster_number, _data in self.clusters.items():
                self.centroids[_cluster_number] = np.mean(
                    convert_to_np_array(
                        _data
                    ),
                    axis = 0
                )
            self.logger.info(f"Centroids Updated")
            
            # Update iteration number
            self.iteration_number += 1
        
        self.logger.info(f"Final Set of Centroids: {self.centroids}")
        self.logger.info(f"Total Number of Iterations: {self.iteration_number}")
        self.logger.info(f"Total Inertia: {self.calculate_inertia()}")

    def get_closest_cluster(
        self,
        centroids: List[np.ndarray],
        x: np.ndarray
    ) -> int:
        return np.argmin([
            euclidean_distance(
                _centroid,
                x
            ) for _centroid in centroids
        ])
    
    def calculate_inertia(self):
        return np.sum([
            np.sum([
                euclidean_distance(
                    self.centroids[_cluster_number],
                    _x
                ) for _x in self.clusters[_cluster_number]
            ]) for _cluster_number in range(self.k)
        ])

    def predict(
        self,
        X: Union[List, pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        X_test = convert_to_np_array(X)
        preds = []
        
        for _x in X_test:
            preds.append(self.get_closest_cluster(list(self.centroids.values()), _x))
        
        return np.array(preds)

# Example Usage
if __name__ == "__main__":
    X_train, X_test, _, _ = load_sample_dataset('classification')
    kmeans = KMeans(k = 3)
    logger = kmeans.logger
    kmeans.fit(X_train)
    y_pred = kmeans.predict(X_test)
    logger.info(f"Predictions: {y_pred.tolist()}")