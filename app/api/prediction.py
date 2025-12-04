"""
Prediction API endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from PIL import Image
import io

from app.models.schemas import PredictionResponse, PredictionRequest, SecondaryPrediction
from ml.inference.predictor import SkinDiseasePredictor
from ml.inference.clinical_predictor import ClinicalSkinPredictor

router = APIRouter()

# Initialize predictors (lazy loading)
_dermo_predictor: SkinDiseasePredictor | None = None
_clinical_predictor: ClinicalSkinPredictor | None = None


def get_dermo_predictor() -> SkinDiseasePredictor:
    """Get or initialize the dermoscopic HAM10000 predictor."""
    global _dermo_predictor
    if _dermo_predictor is None:
        _dermo_predictor = SkinDiseasePredictor()
    return _dermo_predictor


def get_clinical_predictor() -> ClinicalSkinPredictor:
    """Get or initialize the clinical New Dataset 3 predictor."""
    global _clinical_predictor
    if _clinical_predictor is None:
        _clinical_predictor = ClinicalSkinPredictor()
    return _clinical_predictor


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image
    
    Args:
        file: Image file (JPEG, PNG)
    
    Returns:
        Prediction results with class probabilities
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image (JPEG, PNG)"
            )
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Get predictions from both models
        dermo_pred = None
        clinical_pred = None
        warning: str | None = None

        try:
            dermo_pred = get_dermo_predictor().predict(image)
        except Exception as e:  # noqa: BLE001
            warning = f"Dermoscopy model error: {str(e)}"

        try:
            clinical_pred = get_clinical_predictor().predict(image)
        except Exception as e:  # noqa: BLE001
            if warning:
                warning += f" | Clinical model error: {str(e)}"
            else:
                warning = f"Clinical model error: {str(e)}"

        if dermo_pred is None and clinical_pred is None:
            raise HTTPException(
                status_code=500,
                detail="Both models failed to produce a prediction. Please ensure models are trained.",
            )

        # Choose primary prediction based on higher confidence
        if dermo_pred is not None and clinical_pred is not None:
            if dermo_pred["confidence"] >= clinical_pred["confidence"]:
                primary = dermo_pred
                secondary = clinical_pred
                primary_model_type = "dermoscopic"
                secondary_model_type = "clinical"
            else:
                primary = clinical_pred
                secondary = dermo_pred
                primary_model_type = "clinical"
                secondary_model_type = "dermoscopic"
        elif dermo_pred is not None:
            primary = dermo_pred
            secondary = None
            primary_model_type = "dermoscopic"
            secondary_model_type = None
            if warning is None:
                warning = "Clinical model unavailable; only dermoscopic prediction returned."
        else:
            primary = clinical_pred
            secondary = None
            primary_model_type = "clinical"
            secondary_model_type = None
            if warning is None:
                warning = "Dermoscopy model unavailable; only clinical prediction returned."

        secondary_payload = (
            SecondaryPrediction(
                model_type=secondary_model_type,
                predicted_class=secondary["class"],
                confidence=secondary["confidence"],
            )
            if secondary is not None and secondary_model_type is not None
            else None
        )

        return PredictionResponse(
            predicted_class=primary["class"],
            confidence=primary["confidence"],
            probabilities=primary["probabilities"],
            all_classes=primary["all_classes"],
            warning=warning,
            primary_model_type=primary_model_type,
            secondary_prediction=secondary_payload,
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/classes")
async def get_classes():
    """
    Get list of all disease classes
    """
    # For compatibility, expose the original HAM10000 7-class list
    pred = get_dermo_predictor()
    classes = pred.get_class_names()
    return {"classes": classes}


@router.get("/model/info")
async def get_model_info():
    """
    Get model information
    """
    pred = get_dermo_predictor()
    info = pred.get_model_info()
    return info

