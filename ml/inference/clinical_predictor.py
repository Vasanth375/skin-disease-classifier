"""
Clinical model inference for skin disease prediction (New Dataset 3)
"""
import os
from typing import Dict, List

import numpy as np
from PIL import Image
from tensorflow import keras

from app.core.config import settings
from app.utils.image_processing import preprocess_image
import logging

logger = logging.getLogger(__name__)


class ClinicalSkinPredictor:
    """Clinical skin disease prediction using the New Dataset 3 model."""

    CLASS_NAMES = [
        "Acne and Rosacea Photos",
        "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
        "Atopic Dermatitis Photos",
        "Bullous Disease Photos",
        "Cellulitis Impetigo and other Bacterial Infections",
        "Eczema Photos",
        "Exanthems and Drug Eruptions",
        "Hair Loss Photos Alopecia and other Hair Diseases",
        "Herpes HPV and other STDs Photos",
        "Light Diseases and Disorders of Pigmentation",
        "Lupus and other Connective Tissue diseases",
        "Melanoma Skin Cancer Nevi and Moles",
        "Nail Fungus and other Nail Disease",
        "Poison Ivy Photos and other Contact Dermatitis",
        "Psoriasis pictures Lichen Planus and related diseases",
        "Scabies Lyme Disease and other Infestations and Bites",
        "Seborrheic Keratoses and other Benign Tumors",
        "Systemic Disease",
        "Tinea Ringworm Candidiasis and other Fungal Infections",
        "Urticaria Hives",
        "Vascular Tumors",
        "Vasculitis Photos",
        "Warts Molluscum and other Viral Infections",
    ]

    def __init__(self, model_path: str | None = None):
        """
        Initialize clinical predictor.

        Args:
            model_path: Optional explicit path to the clinical model file.
        """
        # Default to settings clinical model path; fall back to hardcoded path if needed
        default_path = settings.CLINICAL_MODEL_PATH
        self.model_path = model_path or default_path
        self.model = None
        self.input_size = settings.MODEL_INPUT_SIZE
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained clinical model."""
        try:
            if os.path.exists(self.model_path):
                logger.info("Loading clinical model from %s", self.model_path)
                self.model = keras.models.load_model(self.model_path)
                logger.info("Clinical model loaded successfully")
            else:
                logger.warning(
                    "Clinical model not found at %s. Please train the clinical model first.",
                    self.model_path,
                )
                self.model = None
        except Exception as e:  # noqa: BLE001
            logger.error("Error loading clinical model: %s", str(e))
            self.model = None

    def predict(self, image: Image.Image) -> Dict:
        """
        Predict clinical skin disease from image.

        Returns a dict with the same keys as SkinDiseasePredictor.predict.
        """
        if self.model is None:
            raise ValueError("Clinical model not loaded. Please train the clinical model first.")

        processed_image = preprocess_image(image, (self.input_size, self.input_size))

        predictions = self.model.predict(processed_image, verbose=0)
        probabilities = predictions[0]

        predicted_idx = int(np.argmax(probabilities))
        predicted_class = self.CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        prob_dict = {
            self.CLASS_NAMES[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": prob_dict,
            "all_classes": self.CLASS_NAMES,
        }

    def get_class_names(self) -> List[str]:
        """Get list of clinical class names."""
        return self.CLASS_NAMES


