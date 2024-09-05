import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Union

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)


def softmax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=1, keepdims=True)


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


def calculate_distance(x: np.ndarray, y: np.ndarray, type: str = 'euclidean') -> np.float64:
    if type == 'euclidean':
        return np.sqrt(np.sum((x - y)**2))
    elif type == 'manhattan':
        return np.sum(np.abs(x - y))
    elif type == 'cosine':
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        raise ValueError('Invalid Distance Type')
    


def get_classification_results(y_true, y_pred) -> dict:
    y_true = convert_to_np_array(y_true)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }


def get_regression_results(y_true, y_pred) -> dict:
    y_true = convert_to_np_array(y_true)
    return {
        'mse': np.mean((y_true - y_pred) ** 2),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'r2': 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    }


def load_sample_dataset(type: str):
    if type == 'classification':
        data = load_iris()
        X = data.data
        y = data.target
        return train_test_split(X, y, test_size=0.2, random_state=42)
    elif type == 'regression':
        data = load_diabetes()
        X = data.data
        y = data.target
        y = y.reshape((len(y), 1))
        return train_test_split(X, y, test_size=0.2, random_state=42)