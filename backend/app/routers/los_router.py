"""
Hospital Length of Stay (LOS) Prediction API.

Layer flow:
    FastAPI Router -> HospitalLOSService -> HospitalLOSImplementation
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.services.los_service import (
    los_service,
    LOSModelError,
    LOSFeatureValidationError,
)

logger = settings.get_logger()

MAX_BATCH_SIZE = int(os.getenv("LOS_MAX_BATCH_SIZE", "500"))
REGISTERED_MODEL_NAME = os.getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "hospital-los-regressor"
)
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

router = APIRouter(
    prefix="/los",
    tags=["Hospital Length of Stay"],
)


class LOSPredictionRequest(BaseModel):
    """Single-record LOS prediction request."""
    features: dict[str, Any] = Field(
        ..., description="Patient and admission features."
    )


class LOSBatchPredictionRequest(BaseModel):
    """Batch LOS prediction request."""
    records: list[dict[str, Any]] = Field(
        ..., description="List of patient feature records."
    )


class LOSPredictionResponse(BaseModel):
    predicted_los_days: float
    unit: str = "days"
    model_name: str
    model_alias: str
    execution_time_seconds: float
    monitoring: dict


class LOSBatchPredictionResponse(BaseModel):
    predictions: list[float]
    unit: str = "days"
    model_name: str
    model_alias: str
    total_records: int
    execution_time_seconds: float
    monitoring: dict


class LOSHealthResponse(BaseModel):
    status: str
    model_name: str
    model_alias: str
    model_uri: str
    tracking_uri: str
    model_version: str | None = None
    signature_available: bool | None = None


def _raise_api_error(exc: Exception) -> None:
    """Translate service-layer exceptions into HTTP responses."""
    if isinstance(exc, LOSFeatureValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if isinstance(exc, LOSModelError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    logger.exception("Unexpected LOS API error.")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Unable to complete Hospital LOS operation.",
    ) from exc


@router.post(
    "/predict",
    response_model=LOSPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Hospital Length of Stay",
)
def predict_los(request: LOSPredictionRequest):
    """POST /los/predict — delegates to service, then implementation."""
    logger.info(
        "LOS single prediction request received | feature_count={}",
        len(request.features),
    )
    try:
        # Router -> Service -> Implementation
        result = los_service.predict_single(request.features)
        return LOSPredictionResponse(**result)
    except Exception as exc:
        _raise_api_error(exc)


@router.post(
    "/predict/batch",
    response_model=LOSBatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch Hospital LOS Prediction",
)
def predict_los_batch(request: LOSBatchPredictionRequest):
    """POST /los/predict/batch — delegates to service, then implementation."""
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch records cannot be empty.",
        )
    if len(request.records) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch size exceeds maximum allowed limit of {MAX_BATCH_SIZE}.",
        )

    logger.info(
        "LOS batch prediction request received | records={}",
        len(request.records),
    )
    try:
        # Router -> Service -> Implementation
        result = los_service.predict_batch(request.records)
        return LOSBatchPredictionResponse(**result)
    except Exception as exc:
        _raise_api_error(exc)


@router.get(
    "/health",
    response_model=LOSHealthResponse,
    summary="Hospital LOS Model Health",
)
def los_model_health():
    """Health check delegates through service to implementation."""
    try:
        # Router -> Service -> Implementation
        result = los_service.health_check()
        return LOSHealthResponse(
            status=result.get("status", "HEALTHY"),
            model_name=result["model_name"],
            model_alias=result["model_alias"],
            model_uri=result["model_uri"],
            tracking_uri=result["tracking_uri"],
            model_version=(
                str(result["model_version"])
                if result.get("model_version") is not None else None
            ),
            signature_available=result.get("signature_available"),
        )
    except Exception as exc:
        if isinstance(exc, LOSFeatureValidationError):
            _raise_api_error(exc)
        if isinstance(exc, LOSModelError):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        logger.exception("LOS model health endpoint failed.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hospital LOS champion model is unavailable.",
        ) from exc
