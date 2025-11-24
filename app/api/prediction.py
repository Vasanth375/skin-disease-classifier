"""
Prediction API endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import numpy as np
from PIL import Image
import io

from app.models.schemas import PredictionResponse, PredictionRequest
from ml.inference.predictor import SkinDiseasePredictor

router = APIRouter()

# Initialize predictor (lazy loading)
predictor = None


def get_predictor():
    """Get or initialize the predictor"""
    global predictor
    if predictor is None:
        predictor = SkinDiseasePredictor()
    return predictor


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
        
        # Get prediction
        pred = get_predictor()
        result = pred.predict(image)
        
        return PredictionResponse(
            predicted_class=result["class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            all_classes=result["all_classes"]
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
    pred = get_predictor()
    classes = pred.get_class_names()
    return {"classes": classes}


@router.get("/model/info")
async def get_model_info():
    """
    Get model information
    """
    pred = get_predictor()
    info = pred.get_model_info()
    return info

