from pathlib import Path
import pandas as pd
import pytest

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data, validate_raw_data


def test_load_raw_data(tmp_path):
    output_file = tmp_path / "iris.csv"
    df = load_raw_data(output_path=str(output_file))

    assert output_file.exists()
    assert df.shape == (150, 6)
    expected_cols = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
        "species",
    }
    assert expected_cols.issubset(set(df.columns))

    # Check 50 samples for each species
    species_counts = df["species"].value_counts().to_dict()
    assert species_counts["Setosa"] == 50
    assert species_counts["Versicolor"] == 50
    assert species_counts["Virginica"] == 50


def test_validate_raw_data_valid(sample_features_list):
    valid_df = pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9],
            "sepal_width": [3.5, 3.0],
            "petal_length": [1.4, 1.4],
            "petal_width": [0.2, 0.2],
            "target": [0, 0],
        }
    )
    # Should not raise exception
    validate_raw_data(valid_df, sample_features_list)


def test_validate_raw_data_missing_column(sample_features_list):
    invalid_df = pd.DataFrame(
        {
            "sepal_length": [5.1],
            "petal_length": [1.4],
            "petal_width": [0.2],
            "target": [0],
        }
    )
    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_raw_data(invalid_df, sample_features_list)


def test_validate_raw_data_nulls(sample_features_list):
    invalid_df = pd.DataFrame(
        {
            "sepal_length": [None],
            "sepal_width": [3.5],
            "petal_length": [1.4],
            "petal_width": [0.2],
            "target": [0],
        }
    )
    with pytest.raises(ValueError, match="contains null or missing"):
        validate_raw_data(invalid_df, sample_features_list)


def test_preprocess_data(tmp_path):
    raw_file = tmp_path / "raw.csv"
    train_file = tmp_path / "train.csv"
    test_file = tmp_path / "test.csv"

    load_raw_data(output_path=str(raw_file))

    train_df, test_df = preprocess_data(
        input_path=str(raw_file),
        train_output_path=str(train_file),
        test_output_path=str(test_file),
        test_size=0.2,
        random_state=42,
        stratify=True,
    )

    assert train_file.exists()
    assert test_file.exists()
    assert len(train_df) == 120
    assert len(test_df) == 30

    # Verify stratified balance in test set (10 each)
    test_counts = test_df["target"].value_counts().to_dict()
    assert test_counts[0] == 10
    assert test_counts[1] == 10
    assert test_counts[2] == 10
