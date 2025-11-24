"""
Image processing utilities
"""
import numpy as np
from PIL import Image
import cv2


def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model input
    
    Args:
        image: PIL Image
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image array
    """
    # Resize image
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array
    img_array = np.array(image)
    
    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Enhance image quality for better prediction
    
    Args:
        image: PIL Image
    
    Returns:
        Enhanced PIL Image
    """
    # Convert to OpenCV format
    img_array = np.array(image)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if len(img_array.shape) == 3:
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(img_array)

