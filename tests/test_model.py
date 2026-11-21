from pathlib import Path
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import evaluate_model
from src.models.predict import IrisPredictor


def test_train_model(trained_artifacts):
    model_path = Path(trained_artifacts["model_path"])
    assert model_path.exists()
    assert model_path.stat().st_size > 0


def test_evaluate_model(trained_artifacts):
    metrics = evaluate_model(
        test_data_path=trained_artifacts["test_path"],
        model_path=trained_artifacts["model_path"],
        metrics_path=trained_artifacts["metrics_path"],
        confusion_matrix_path=trained_artifacts["cm_path"],
        tracking_uri=trained_artifacts["tracking_uri"],
    )

    assert Path(trained_artifacts["metrics_path"]).exists()
    assert Path(trained_artifacts["cm_path"]).exists()

    assert "accuracy" in metrics
    assert "f1_weighted" in metrics
    # High standard benchmark for Iris classification
    assert metrics["accuracy"] >= 0.90


def test_iris_predictor_setosa(trained_artifacts, sample_setosa):
    predictor = IrisPredictor(model_path=trained_artifacts["model_path"])
    result = predictor.predict(sample_setosa)

    assert result["predicted_species"] == "Setosa"
    assert result["predicted_class_id"] == 0
    assert result["confidence"] > 0.8
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, rel=1e-3)


def test_iris_predictor_versicolor(trained_artifacts, sample_versicolor):
    predictor = IrisPredictor(model_path=trained_artifacts["model_path"])
    result = predictor.predict(sample_versicolor)

    assert result["predicted_species"] == "Versicolor"
    assert result["predicted_class_id"] == 1
    assert result["confidence"] > 0.5


def test_iris_predictor_virginica(trained_artifacts, sample_virginica):
    predictor = IrisPredictor(model_path=trained_artifacts["model_path"])
    result = predictor.predict(sample_virginica)

    assert result["predicted_species"] == "Virginica"
    assert result["predicted_class_id"] == 2
    assert result["confidence"] > 0.5


def test_iris_predictor_missing_keys(trained_artifacts):
    predictor = IrisPredictor(model_path=trained_artifacts["model_path"])
    with pytest.raises(ValueError, match="Missing required feature keys"):
        predictor.predict({"sepal_length": 5.1})
