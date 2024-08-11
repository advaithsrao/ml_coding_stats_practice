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
        logger.info(f"Time taken: {time.time() - start}")
        logger.info("----- Training complete -----")
        return result
    return wrapper