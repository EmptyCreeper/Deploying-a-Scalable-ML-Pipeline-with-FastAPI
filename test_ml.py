import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model

CAT_FEATURES = ["sex"]


@pytest.fixture
def sample_data():
    """Small, dummy dataset for use in test functions. Salary is >50K when age >= 40."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 45, 50, 55],
            "hours-per-week": [40, 35, 40, 45, 50, 40],
            "sex": ["Male", "Female", "Male", "Female", "Male", "Female"],
            "salary": ["<=50K", "<=50K", "<=50K", ">50K", ">50K", ">50K"],
        }
    )


def test_process_data_output_shapes(sample_data):
    """
    Test to see if data is in the expected shape after being put through process_data.
    """
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    assert X.shape == (6, 4)
    assert y.shape == (6,)
    assert list(y) == [0, 0, 0, 1, 1, 1]


def test_train_model_and_inference(sample_data):
    """
    test to see if the returned model is the correct type, has one prediction per row, and 
    and that every prediction is correct
    """
    X, y, _, _ = process_data(
        sample_data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(model, RandomForestClassifier)
    assert len(preds) == len(y)
    assert np.array_equal(preds, y)


def test_compute_model_metrics_known_values():
    """
    test to make sure compute_model_metrics gives the right precision, recall and F1 for a small
    example with one true positive, no false positives, one false negative.
    """
    y = np.array([1, 1, 0, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(2 / 3)
