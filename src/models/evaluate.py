import argparse
import json
import os
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.config import load_config
from src.utils.logger import get_logger

# Ensure backwards compatibility with file store if used
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

logger = get_logger("evaluate")


def evaluate_model(
    test_data_path: str = "data/processed/test.csv",
    model_path: str = "models/random_forest_model.joblib",
    metrics_path: str = "metrics/scores.json",
    confusion_matrix_path: str = "metrics/confusion_matrix.png",
    features: list[str] = None,
    target: str = "target",
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "iris-classification-rf",
) -> dict:
    """Evaluates trained model on test data, logs metrics & plots."""
    if features is None:
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    logger.info(f"Loading test data from {test_data_path}...")
    test_df = pd.read_csv(test_data_path)
    X_test = test_df[features]
    y_test = test_df[target]

    logger.info(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted"))
    recall = float(recall_score(y_test, y_pred, average="weighted"))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    species_names = ["Setosa", "Versicolor", "Virginica"]
    report = classification_report(
        y_test, y_pred, target_names=species_names, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report,
    }

    # Save metrics JSON
    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {metrics_file}: {metrics}")

    # Plot and save confusion matrix
    cm_file = Path(confusion_matrix_path)
    cm_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
    fig.colorbar(cax)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                x=j,
                y=i,
                s=str(cm[i, j]),
                va="center",
                ha="center",
                size="large",
                weight="bold",
            )

    ax.set_xticks(range(len(species_names)))
    ax.set_yticks(range(len(species_names)))
    ax.set_xticklabels(species_names)
    ax.set_yticklabels(species_names)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("Iris Classification Confusion Matrix", pad=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(cm_file, dpi=200)
    plt.close(fig)
    logger.info(f"Confusion matrix plot saved to {cm_file}")

    # Log to MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="evaluate_model"):
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_artifact(str(metrics_file))
        mlflow.log_artifact(str(cm_file))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Iris classification model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/params.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    prep_cfg = config.get("preprocess", {})
    train_cfg = config.get("train", {})
    eval_cfg = config.get("evaluate", {})
    mlflow_cfg = config.get("mlflow", {})

    test_data_path = prep_cfg.get("test_data_path", "data/processed/test.csv")
    model_path = train_cfg.get("model_path", "models/random_forest_model.joblib")
    metrics_path = eval_cfg.get("metrics_path", "metrics/scores.json")
    confusion_matrix_path = eval_cfg.get(
        "confusion_matrix_path", "metrics/confusion_matrix.png"
    )
    features = prep_cfg.get(
        "features", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    target = prep_cfg.get("target", "target")
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "iris-classification-rf")

    evaluate_model(
        test_data_path=test_data_path,
        model_path=model_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        features=features,
        target=target,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    main()
