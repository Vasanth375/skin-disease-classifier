"""
Model inference for skin disease prediction
"""
import os
import numpy as np
from tensorflow import keras
from PIL import Image
from typing import Dict, List
import logging

from app.core.config import settings
from app.utils.image_processing import preprocess_image

logger = logging.getLogger(__name__)


class SkinDiseasePredictor:
    """Skin disease prediction class"""
    
    # HAM10000 class names
    CLASS_NAMES = [
        "actinic_keratosis",
        "basal_cell_carcinoma",
        "benign_keratosis",
        "dermatofibroma",
        "melanoma",
        "nevus",
        "vascular_lesion"
    ]
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model file
        """
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self.input_size = settings.MODEL_INPUT_SIZE
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading model from {self.model_path}")
                self.model = keras.models.load_model(self.model_path)
                logger.info("Model loaded successfully")
            else:
                logger.warning(f"Model not found at {self.model_path}. Please train the model first.")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict skin disease from image
        
        Args:
            image: PIL Image
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        # Preprocess image
        processed_image = preprocess_image(image, (self.input_size, self.input_size))
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        probabilities = predictions[0]
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # Create probabilities dictionary
        prob_dict = {
            self.CLASS_NAMES[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "all_classes": self.CLASS_NAMES
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.CLASS_NAMES
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        if self.model is None:
            return {
                "status": "not_loaded",
                "message": "Model not loaded"
            }
        
        return {
            "status": "loaded",
            "input_size": self.input_size,
            "num_classes": len(self.CLASS_NAMES),
            "classes": self.CLASS_NAMES,
            "model_path": self.model_path
        }

