from sklearn.linear_model import LogisticRegression
from ml.helpers import load_sample_dataset
import pickle
from ml import logger
import numpy as np

# training
X_train, X_test, y_train, y_test = load_sample_dataset('classification')
X = np.concatenate((X_train, X_test))
y = np.concatenate((y_train, y_test))
model = LogisticRegression()
model.fit(X, y)
logger.info('Model Trained Successfully')

# save model as pickle
with open('./sample_ml_model.pkl', 'wb') as f:
    pickle.dump(model, f)

logger.info('Model Saved Successfully')