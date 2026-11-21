import argparse
import os
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.utils.config import load_config
from src.utils.logger import get_logger

# Ensure backwards compatibility with file store if used
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

logger = get_logger("train")


def train_model(
    train_data_path: str = "data/processed/train.csv",
    model_output_path: str = "models/random_forest_model.joblib",
    features: list[str] = None,
    target: str = "target",
    n_estimators: int = 100,
    max_depth: int = 4,
    min_samples_split: int = 2,
    criterion: str = "gini",
    random_state: int = 42,
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "iris-classification-rf",
) -> RandomForestClassifier:
    """Trains a Random Forest classifier and tracks parameters with MLflow.

    Args:
        train_data_path: Path to processed train CSV.
        model_output_path: Path where serialized model will be saved.
        features: List of feature names.
        target: Name of target column.
        n_estimators: Number of trees.
        max_depth: Max depth of tree.
        min_samples_split: Minimum samples required to split an internal node.
        criterion: Impurity function ('gini' or 'entropy').
        random_state: Seed for reproducibility.
        tracking_uri: MLflow tracking directory or server URI.
        experiment_name: MLflow experiment name.

    Returns:
        RandomForestClassifier: Trained model.
    """
    if features is None:
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    logger.info(f"Loading training data from {train_data_path}...")
    train_df = pd.read_csv(train_data_path)

    X_train = train_df[features]
    y_train = train_df[target]

    logger.info(
        f"Training RandomForestClassifier with {len(X_train)} samples, "
        f"n_estimators={n_estimators}, max_depth={max_depth}, criterion={criterion}..."
    )

    # Initialize model
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state,
    )

    # MLflow tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="train_random_forest") as run:
        # Log hyperparameters
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "criterion": criterion,
                "random_state": random_state,
                "features": features,
                "n_train_samples": len(X_train),
            }
        )

        # Fit model
        rf.fit(X_train, y_train)

        # Training score
        train_accuracy = float(rf.score(X_train, y_train))
        mlflow.log_metric("train_accuracy", train_accuracy)
        logger.info(f"Training accuracy: {train_accuracy:.4f}")

        # Ensure output model directory exists
        dest_path = Path(model_output_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model with joblib
        joblib.dump(rf, dest_path)
        logger.info(f"Model saved to {dest_path}")

        # Log model artifact to MLflow using cloudpickle format
        mlflow.sklearn.log_model(
            sk_model=rf,
            name="iris_rf_model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            input_example=X_train.iloc[:2],
        )
        logger.info(f"MLflow Run ID: {run.info.run_id}")

    return rf


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest on Iris dataset.")
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
    mlflow_cfg = config.get("mlflow", {})
    base_cfg = config.get("base", {})

    train_data_path = prep_cfg.get("train_data_path", "data/processed/train.csv")
    model_output_path = train_cfg.get(
        "model_path", "models/random_forest_model.joblib"
    )
    features = prep_cfg.get(
        "features", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    target = prep_cfg.get("target", "target")

    n_estimators = int(train_cfg.get("n_estimators", 100))
    max_depth = int(train_cfg.get("max_depth", 4))
    min_samples_split = int(train_cfg.get("min_samples_split", 2))
    criterion = str(train_cfg.get("criterion", "gini"))
    random_state = int(
        train_cfg.get("random_state", base_cfg.get("random_state", 42))
    )

    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "iris-classification-rf")

    train_model(
        train_data_path=train_data_path,
        model_output_path=model_output_path,
        features=features,
        target=target,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )


if __name__ == "__main__":
    main()
