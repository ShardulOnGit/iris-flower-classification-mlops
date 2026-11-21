import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("load_data")


def load_raw_data(output_path: str = "data/raw/iris.csv") -> pd.DataFrame:
    """Loads Iris dataset from scikit-learn and saves it to CSV.

    Args:
        output_path: Destination path for raw CSV data.

    Returns:
        pd.DataFrame: Loaded DataFrame with standardized column names.
    """
    logger.info("Loading Iris dataset from scikit-learn...")
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    # Standardize column names to snake_case without units
    column_mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        "target": "target",
    }
    df = df.rename(columns=column_mapping)

    # Add species name column for human readability
    species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    df["species"] = df["target"].map(species_map)

    # Ensure output directory exists
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(destination, index=False)
    logger.info(
        f"Saved raw data to {destination} with shape {df.shape}. "
        f"Class distribution: {df['species'].value_counts().to_dict()}"
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Load raw Iris dataset.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/params.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = config.get("data_load", {}).get("raw_data_path", "data/raw/iris.csv")
    load_raw_data(output_path=output_path)


if __name__ == "__main__":
    main()
