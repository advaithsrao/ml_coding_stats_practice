import pytest
from ml.classification.knn import KNN
from ml.helpers import load_sample_dataset, get_classification_results

@pytest.fixture
def sample_data():
    return load_sample_dataset('classification')

def test_knn(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    knn = KNN(k=1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    results = get_classification_results(y_test, preds)
    assert results['accuracy'] > 0.9
    assert results['precision'] > 0.9
    assert results['recall'] > 0.9
    assert results['f1'] > 0.9