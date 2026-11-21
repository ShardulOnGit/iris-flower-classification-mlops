from pathlib import Path
from typing import Any, Dict, List, Union
import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("predict")

SPECIES_MAP = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
FEATURE_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


class IrisPredictor:
    """Inference engine for Iris flower classification."""

    def __init__(self, model_path: Union[str, Path] = "models/random_forest_model.joblib"):
        self.model_path = Path(model_path)
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Loads serialized model from disk."""
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Please train the model first."
            )
        self.model = joblib.load(self.model_path)
        logger.info(f"Loaded model successfully from {self.model_path}")

    def predict(self, features: Union[Dict[str, float], List[float], pd.DataFrame]) -> Dict[str, Any]:
        """Predicts species for a single iris flower observation.

        Args:
            features: Dictionary, list of 4 floats, or DataFrame with feature measurements.

        Returns:
            Dict containing predicted class ID, species name, and class probabilities.
        """
        if self.model is None:
            self.load_model()

        if isinstance(features, dict):
            # Verify required keys
            missing = [f for f in FEATURE_NAMES if f not in features]
            if missing:
                raise ValueError(f"Missing required feature keys: {missing}")
            input_df = pd.DataFrame([[features[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        elif isinstance(features, (list, tuple)):
            if len(features) != 4:
                raise ValueError(f"Expected 4 feature values, got {len(features)}")
            input_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        elif isinstance(features, pd.DataFrame):
            input_df = features[FEATURE_NAMES]
        else:
            raise TypeError("Unsupported input type for features.")

        pred_class_id = int(self.model.predict(input_df)[0])
        probabilities = self.model.predict_proba(input_df)[0]

        species = SPECIES_MAP.get(pred_class_id, "Unknown")
        prob_dict = {SPECIES_MAP[i]: float(prob) for i, prob in enumerate(probabilities)}

        return {
            "predicted_class_id": pred_class_id,
            "predicted_species": species,
            "confidence": float(np.max(probabilities)),
            "probabilities": prob_dict,
        }

    def predict_batch(self, samples: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Predicts species for a batch of flower observations."""
        return [self.predict(sample) for sample in samples]
