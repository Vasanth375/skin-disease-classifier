"""
Image processing utilities
"""
import numpy as np
from PIL import Image


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

