import os
import pytest
from pathlib import Path

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.models.train import train_model

# Ensure backwards compatibility
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


@pytest.fixture
def sample_setosa():
    return {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }


@pytest.fixture
def sample_versicolor():
    return {
        "sepal_length": 6.0,
        "sepal_width": 2.9,
        "petal_length": 4.5,
        "petal_width": 1.5,
    }


@pytest.fixture
def sample_virginica():
    return {
        "sepal_length": 6.9,
        "sepal_width": 3.1,
        "petal_length": 5.4,
        "petal_width": 2.1,
    }


@pytest.fixture
def sample_features_list():
    return ["sepal_length", "sepal_width", "petal_length", "petal_width"]


@pytest.fixture(scope="session")
def trained_artifacts(tmp_path_factory):
    """Fixture that prepares data and trains a model once for all test sessions."""
    temp_dir = tmp_path_factory.mktemp("ml_artifacts")
    raw_path = temp_dir / "iris.csv"
    train_path = temp_dir / "train.csv"
    test_path = temp_dir / "test.csv"
    model_path = temp_dir / "rf_model.joblib"
    metrics_path = temp_dir / "scores.json"
    cm_path = temp_dir / "cm.png"
    tracking_db = temp_dir / "mlflow_test.db"
    tracking_uri = f"sqlite:///{tracking_db.as_posix()}"

    load_raw_data(output_path=str(raw_path))
    preprocess_data(
        input_path=str(raw_path),
        train_output_path=str(train_path),
        test_output_path=str(test_path),
        random_state=42,
    )
    train_model(
        train_data_path=str(train_path),
        model_output_path=str(model_path),
        tracking_uri=tracking_uri,
        random_state=42,
    )

    return {
        "temp_dir": str(temp_dir),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "cm_path": str(cm_path),
        "tracking_uri": tracking_uri,
    }
