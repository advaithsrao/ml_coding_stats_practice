import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Union


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def convert_to_np_array(x: Union[List, pd.DataFrame, np.ndarray]) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        try:
            return np.array(x.values.tolist())
        except Exception as ex:
            raise ex
    elif isinstance(x, list):
        try:
            return np.array(x)
        except Exception as ex:
            raise ex
    else:
        return x


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.float64:
    return np.sqrt(np.sum((x - y)**2))


def get_classification_results(y_true, y_pred) -> dict:
    y_true = convert_to_np_array(y_true)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }


def load_sample_dataset(type: str):
    if type == 'classification':
        data = load_iris()
        X = data.data
        y = data.target
        return train_test_split(X, y, test_size=0.2, random_state=42)