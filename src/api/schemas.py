from typing import Dict, List
from pydantic import BaseModel, Field


class IrisInput(BaseModel):
    """Schema for flower measurements input (in centimeters)."""

    sepal_length: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Sepal length in centimeters",
        examples=[5.1],
    )
    sepal_width: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Sepal width in centimeters",
        examples=[3.5],
    )
    petal_length: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Petal length in centimeters",
        examples=[1.4],
    )
    petal_width: float = Field(
        ...,
        gt=0.0,
        le=20.0,
        description="Petal width in centimeters",
        examples=[0.2],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    }


class BatchIrisInput(BaseModel):
    """Schema for batch prediction input."""

    flowers: List[IrisInput] = Field(
        ...,
        min_length=1,
        description="List of flower measurement objects",
    )


class PredictionOutput(BaseModel):
    """Schema for single prediction output."""

    predicted_class_id: int = Field(..., description="Integer class identifier (0, 1, or 2)")
    predicted_species: str = Field(..., description="Predicted flower species: Setosa, Versicolor, or Virginica")
    confidence: float = Field(..., description="Probability of predicted class (0.0 to 1.0)")
    probabilities: Dict[str, float] = Field(
        ..., description="Probability distribution across all species classes"
    )


class BatchPredictionOutput(BaseModel):
    """Schema for batch prediction output."""

    count: int = Field(..., description="Number of items predicted")
    predictions: List[PredictionOutput] = Field(..., description="List of predictions")


class HealthResponse(BaseModel):
    """Schema for health status endpoint."""

    status: str = Field(..., description="Service health state")
    model_loaded: bool = Field(..., description="Indicates if ML model is active in memory")
    version: str = Field(..., description="API Version")


class ModelInfoResponse(BaseModel):
    """Schema for model metadata and description."""

    project_name: str
    algorithm: str
    target_classes: List[str]
    input_features: List[str]
    model_path: str
