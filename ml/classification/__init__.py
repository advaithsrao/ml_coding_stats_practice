from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

# Supervised Classifier Base Class
class SupervisedClassifier(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def fit(
        self,
        X: list | pd.DataFrame | np.ndarray,
        y: list | pd.DataFrame | np.ndarray,
        *args,
        **kwargs
    ):
        pass

    @abstractmethod
    def predict(
        self, 
        X: list | pd.DataFrame | np.ndarray,
        *args,
        **kwargs
    ):
        pass


# Unsupervised Classifier Base Class
class UnsupervisedClassifier(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def fit(
        self,
        X: list | pd.DataFrame | np.ndarray,
        *args,
        **kwargs
    ):
        pass

    @abstractmethod
    def predict(
        self,
        X: list | pd.DataFrame | np.ndarray,
        *args,
        **kwargs
    ):
        pass
