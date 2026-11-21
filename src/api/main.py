import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BatchIrisInput,
    BatchPredictionOutput,
    HealthResponse,
    IrisInput,
    ModelInfoResponse,
    PredictionOutput,
)
from src.models.predict import IrisPredictor, SPECIES_MAP, FEATURE_NAMES
from src.utils.logger import get_logger

logger = get_logger("fastapi_app")

predictor: Optional[IrisPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading ML models on startup."""
    global predictor
    model_path = os.getenv("MODEL_PATH", "models/random_forest_model.joblib")
    if predictor is None:
        try:
            # If model artifact does not exist (e.g. fresh container without DVC pull), auto-train it
            if not Path(model_path).is_file():
                logger.info(f"Model file not found at {model_path}. Auto-training model for container startup...")
                from src.data.load_data import load_raw_data
                from src.data.preprocess import preprocess_data
                from src.models.train import train_model

                load_raw_data()
                preprocess_data()
                train_model(model_output_path=model_path, tracking_uri="sqlite:///mlflow.db")

            predictor = IrisPredictor(model_path=model_path)
            logger.info("Random Forest Model loaded successfully into memory.")
        except Exception as exc:
            logger.warning(f"Model could not be loaded at startup from {model_path}: {exc}")
            predictor = None
    yield
    logger.info("Shutting down API service.")


app = FastAPI(
    title="Iris Flower Classification MLOps API",
    description=(
        "Production-grade RESTful API for classifying Iris flowers (Setosa, Versicolor, "
        "Virginica) using a Random Forest Classifier with MLOps tracking and validation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for frontend / external integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with API details."""
    return {
        "message": "Welcome to Iris Flower Classification MLOps Service",
        "documentation": "/docs",
        "version": "0.1.0",
        "algorithm": "Random Forest Classifier",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns service health status and model availability."""
    is_loaded = predictor is not None and predictor.model is not None
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        version="0.1.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model Info"])
async def model_info():
    """Returns metadata regarding the trained model and features."""
    return ModelInfoResponse(
        project_name="Iris Flower Classification using MLOps",
        algorithm="Random Forest Classifier (scikit-learn)",
        target_classes=[SPECIES_MAP[i] for i in sorted(SPECIES_MAP.keys())],
        input_features=FEATURE_NAMES,
        model_path=str(predictor.model_path if predictor else "models/random_forest_model.joblib"),
    )


@app.post(
    "/predict",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_single(iris: IrisInput):
    """Predict the flower species for a single iris flower observation."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train or provide model artifact first.",
        )

    try:
        input_dict = iris.model_dump()
        result = predictor.predict(input_dict)
        return PredictionOutput(**result)
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference failed: {str(exc)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict_batch(batch: BatchIrisInput):
    """Predict flower species for a batch of iris flowers."""
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please train or provide model artifact first.",
        )

    try:
        samples = [item.model_dump() for item in batch.flowers]
        results = predictor.predict_batch(samples)
        predictions = [PredictionOutput(**res) for res in results]
        return BatchPredictionOutput(count=len(predictions), predictions=predictions)
    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch inference failed: {str(exc)}",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)
