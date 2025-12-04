"""
Pydantic schemas for prediction API.
"""
from typing import Dict, Optional

from pydantic import BaseModel


class SecondaryPrediction(BaseModel):
    """Optional secondary model prediction info."""

    model_type: str
    predicted_class: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response returned by /predict endpoint."""

    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    all_classes: list[str]
    warning: Optional[str] = None
    # Extra fields for dual-model setup
    primary_model_type: Optional[str] = None
    secondary_prediction: Optional[SecondaryPrediction] = None


