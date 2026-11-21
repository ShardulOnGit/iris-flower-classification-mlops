import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("preprocess")


def validate_raw_data(df: pd.DataFrame, expected_features: list[str]) -> None:
    """Validates raw data integrity and structure."""
    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    if "target" not in df.columns:
        raise ValueError("Target column is missing from dataset.")

    if df[expected_features].isnull().any().any():
        raise ValueError("Dataset contains null or missing values.")

    # Check for non-negative values in measurements
    if (df[expected_features] <= 0).any().any():
        raise ValueError("Measurement features must be strictly positive numbers.")


def preprocess_data(
    input_path: str = "data/raw/iris.csv",
    train_output_path: str = "data/processed/train.csv",
    test_output_path: str = "data/processed/test.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    features: list[str] = None,
    target: str = "target",
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits raw dataset into stratified train and test sets."""
    if features is None:
        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    logger.info(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)

    validate_raw_data(df, features)

    stratify_col = df[target] if stratify else None
    logger.info(
        f"Splitting data with test_size={test_size}, random_state={random_state}, stratify={stratify}..."
    )
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    # Ensure output directories exist
    train_path = Path(train_output_path)
    test_path = Path(test_output_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        f"Saved train data ({train_df.shape[0]} rows) to {train_path} and "
        f"test data ({test_df.shape[0]} rows) to {test_path}"
    )
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split Iris dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/params.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    prep_cfg = config.get("preprocess", {})
    base_cfg = config.get("base", {})

    input_path = config.get("data_load", {}).get("raw_data_path", "data/raw/iris.csv")
    train_output = prep_cfg.get("train_data_path", "data/processed/train.csv")
    test_output = prep_cfg.get("test_data_path", "data/processed/test.csv")
    test_size = float(prep_cfg.get("test_size", 0.2))
    random_state = int(base_cfg.get("random_state", 42))
    features = prep_cfg.get(
        "features", ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )
    target = prep_cfg.get("target", "target")
    stratify = bool(prep_cfg.get("stratify", True))

    preprocess_data(
        input_path=input_path,
        train_output_path=train_output,
        test_output_path=test_output,
        test_size=test_size,
        random_state=random_state,
        features=features,
        target=target,
        stratify=stratify,
    )


if __name__ == "__main__":
    main()
