from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Union
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def time_ml_training(func): 
    def wrapper(*args,**kwargs):
        start = time.time()
        logger.info("----- Training started -----")
        result = func(*args,**kwargs)
        logger.info(f"Time taken for training: {time.time() - start}")
        logger.info("----- Training complete -----")
        return result
    return wrapper

def time_operation(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"*** Operation Name: {func.__name__} ***")
        result = func(*args, **kwargs)
        logger.info(f"Time taken for operation: {time.time() - start}")
        logger.info("*** Operation Complete ***")
        return result
    return wrapper


class GradientDescent(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def update_param(
        self,
        param: float,
        *args,
        **kwargs
    ):
        pass
