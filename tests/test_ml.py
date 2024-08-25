import pytest

from ml.classification.knn import KNN
from ml.classification.kmeans import KMeans
from ml.helpers import load_sample_dataset, get_classification_results

@pytest.fixture
def sample_data():
    return load_sample_dataset('classification')

def test_knn(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    knn = KNN(k=1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)

    #Check fns
    assert knn.__getattribute__('fit')
    assert knn.__getattribute__('predict')
    assert knn.k == 1

    #Results 
    results = get_classification_results(y_test, preds)
    assert results['accuracy'] > 0.9
    assert results['precision'] > 0.9
    assert results['recall'] > 0.9
    assert results['f1'] > 0.9

def test_kmeans(sample_data):
    X_train, X_test, _, _ = sample_data
    kmeans = KMeans(k=3)
    kmeans.fit(X_train)
    preds = kmeans.predict(X_test)

    #Check fns
    assert kmeans.__getattribute__('fit')
    assert kmeans.__getattribute__('predict')
    assert kmeans.k == 3

    #Results
    assert len(kmeans.centroids) == 3
    assert len(kmeans.clusters) == 3
    assert len(preds) == len(X_test)
    assert kmeans.calculate_inertia() > 0