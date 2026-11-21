import pytest
from fastapi.testclient import TestClient

from src.api.main import app
import src.api.main as main_module
from src.models.predict import IrisPredictor


@pytest.fixture(scope="module")
def client(trained_artifacts):
    main_module.predictor = IrisPredictor(model_path=trained_artifacts["model_path"])
    with TestClient(app) as test_client:
        yield test_client
    main_module.predictor = None


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "documentation" in data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "Setosa" in data["target_classes"]
    assert "sepal_length" in data["input_features"]


def test_predict_endpoint_success(client, sample_setosa):
    response = client.post("/predict", json=sample_setosa)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_species"] == "Setosa"
    assert data["predicted_class_id"] == 0
    assert data["confidence"] > 0.8
    assert "Setosa" in data["probabilities"]


def test_predict_endpoint_validation_negative(client):
    payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error (gt=0.0)


def test_predict_endpoint_validation_missing_field(client):
    payload = {
        "sepal_length": 5.1,
        "petal_length": 1.4,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_endpoint_success(
    client, sample_setosa, sample_versicolor, sample_virginica
):
    payload = {
        "flowers": [
            sample_setosa,
            sample_versicolor,
            sample_virginica,
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    species_results = [p["predicted_species"] for p in data["predictions"]]
    assert species_results == ["Setosa", "Versicolor", "Virginica"]
