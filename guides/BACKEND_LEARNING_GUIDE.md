# Backend Learning Guide - Skin Disease Classifier

Welcome! This guide will help you understand the backend codebase step by step, even if you're new to the technologies used.

---

## 📚 Table of Contents

1. [Tech Stack Overview](#tech-stack-overview)
2. [Project Structure](#project-structure)
3. [Learning Path](#learning-path)
4. [Core Concepts Explained](#core-concepts-explained)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [How It All Works Together](#how-it-all-works-together)
7. [Hands-On Exercises](#hands-on-exercises)
8. [Resources for Learning](#resources-for-learning)

---

## 🛠️ Tech Stack Overview

### What Technologies Are Used?

1. **Python** - The programming language
2. **FastAPI** - Web framework for building APIs (like Express.js for Node.js)
3. **TensorFlow/Keras** - Machine learning library for neural networks
4. **PIL (Pillow)** - Image processing library
5. **Pydantic** - Data validation library
6. **Uvicorn** - ASGI server to run FastAPI

### Why These Technologies?

- **FastAPI**: Fast, modern, automatic API documentation
- **TensorFlow**: Industry-standard for deep learning
- **Pydantic**: Ensures data is valid before processing

---

## 📁 Project Structure

```
skin-disease-classifier/
├── app/                          # Main application code
│   ├── __init__.py              # Makes it a Python package
│   ├── main.py                  # Application entry point
│   ├── api/                     # API endpoints
│   │   ├── __init__.py
│   │   └── prediction.py            # Prediction endpoint
│   ├── core/                    # Configuration
│   │   ├── __init__.py
│   │   └── config.py              # Settings and configuration
│   ├── models/                   # Data models/schemas
│   │   ├── __init__.py
│   │   └── schemas.py             # Request/Response data structures
│   └── utils/                    # Helper functions
│       ├── __init__.py
│       └── image_processing.py    # Image preprocessing
│
├── ml/                           # Machine learning code
│   ├── inference/                # Model prediction code
│   │   ├── predictor.py           # Dermoscopic model predictor
│   │   └── clinical_predictor.py  # Clinical model predictor
│   └── training/                 # Model training code
│       ├── train_model.py         # Train HAM10000 model
│       └── train_clinical_model.py # Train clinical model
│
├── models/                       # Saved trained models (.h5 files)
├── data/                         # Dataset folders
├── run.py                        # Script to start the server
└── requirements.txt              # Python dependencies
```

---

## 🎯 Learning Path

### Step 1: Understand Python Basics (If Needed)
- Variables, functions, classes
- Import statements
- Error handling (try/except)

### Step 2: Learn FastAPI Basics
- What is an API?
- HTTP methods (GET, POST)
- Request/Response

### Step 3: Understand Our Application
- Start with `main.py` (simplest)
- Then `config.py` (configuration)
- Then `schemas.py` (data structures)
- Then `prediction.py` (main logic)
- Finally ML code (most complex)

---

## 💡 Core Concepts Explained

### 1. **API Endpoint**
An API endpoint is a URL where your application can receive requests.

**Example:**
- URL: `http://localhost:8000/api/v1/predict`
- Method: POST
- Purpose: Upload image and get prediction

### 2. **Request/Response**
- **Request**: Data sent TO the server (e.g., image file)
- **Response**: Data sent BACK from server (e.g., prediction result)

### 3. **Model (ML)**
A trained neural network that can recognize patterns in images.

### 4. **Predictor**
Code that loads a model and uses it to make predictions.

---

## 📄 File-by-File Breakdown

### **1. `run.py` - Starting Point** ⭐ START HERE

**What it does:** Starts the web server

```python
import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",           # Location of FastAPI app
        host=settings.API_HOST,   # Server address (localhost)
        port=settings.API_PORT,    # Port number (8000)
        reload=settings.API_RELOAD # Auto-reload on code changes
    )
```

**Key Concepts:**
- `uvicorn.run()` - Starts the server
- `"app.main:app"` - Tells uvicorn where to find the FastAPI app
- Settings come from `config.py`

**To Run:**
```bash
python run.py
```

---

### **2. `app/main.py` - Application Setup**

**What it does:** Creates the FastAPI application and sets up routes

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import prediction

# Create FastAPI app
app = FastAPI(
    title="Skin Disease Classifier API",
    description="Real-time skin disease classification using CNN",
    version="1.0.0"
)

# Enable CORS (allows frontend to call API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction routes
app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])
```

**Key Concepts:**
- `FastAPI()` - Creates the application
- `CORS` - Allows frontend (React) to call backend
- `include_router()` - Adds API endpoints from `prediction.py`

**Endpoints Created:**
- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/v1/predict` - Main prediction endpoint
- `GET /api/v1/classes` - Get class list
- `GET /api/v1/model/info` - Get model info

---

### **3. `app/core/config.py` - Configuration**

**What it does:** Stores all application settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    
    # Model Configuration
    MODEL_PATH: str = "./models/skin_disease_model.h5"
    MODEL_INPUT_SIZE: int = 224  # Image size for model
    
    # Data Configuration
    DATA_DIR: str = "./data"
    PROCESSED_DATA_DIR: str = "./data/processed"

settings = Settings()  # Create instance
```

**Key Concepts:**
- `BaseSettings` - Pydantic class for settings
- All settings in one place (easy to change)
- Can be overridden with environment variables

**Usage:**
```python
from app.core.config import settings
print(settings.API_PORT)  # 8000
```

---

### **4. `app/models/schemas.py` - Data Structures**

**What it does:** Defines the structure of request/response data

```python
from pydantic import BaseModel

class SecondaryPrediction(BaseModel):
    """Info about the other model's prediction"""
    model_type: str           # "dermoscopic" or "clinical"
    predicted_class: str       # Disease name
    confidence: float         # 0.0 to 1.0

class PredictionResponse(BaseModel):
    """Response sent back to frontend"""
    predicted_class: str                    # Main prediction
    confidence: float                       # Confidence score
    probabilities: Dict[str, float]         # All class probabilities
    all_classes: list[str]                  # List of all possible classes
    warning: Optional[str] = None          # Warning message (if any)
    primary_model_type: Optional[str] = None # Which model was used
    secondary_prediction: Optional[SecondaryPrediction] = None
```

**Key Concepts:**
- `BaseModel` - Pydantic class for data validation
- Ensures data is correct format before processing
- Auto-generates API documentation

**Example Response:**
```json
{
  "predicted_class": "melanoma",
  "confidence": 0.95,
  "probabilities": {
    "melanoma": 0.95,
    "nevus": 0.03,
    ...
  },
  "all_classes": ["actinic_keratosis", "melanoma", ...],
  "primary_model_type": "dermoscopic",
  "secondary_prediction": {
    "model_type": "clinical",
    "predicted_class": "Melanoma Skin Cancer Nevi and Moles",
    "confidence": 0.87
  }
}
```

---

### **5. `app/utils/image_processing.py` - Image Preprocessing**

**What it does:** Prepares images for the ML model

```python
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Convert image to format model expects
    
    Steps:
    1. Resize to 224x224 (model requirement)
    2. Convert to numpy array
    3. Normalize pixels: [0, 255] → [0.0, 1.0]
    4. Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    """
    # Resize
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to array
    img_array = np.array(image)  # Shape: (224, 224, 3)
    
    # Normalize
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    
    return img_array
```

**Key Concepts:**
- Models need specific input format
- Images must be 224x224 pixels
- Pixel values must be 0.0-1.0 (not 0-255)
- Batch dimension needed for TensorFlow

**Why?**
- Neural networks expect consistent input size
- Normalization helps training/prediction
- Batch dimension allows processing multiple images

---

### **6. `app/api/prediction.py` - Main API Logic** ⭐ MOST IMPORTANT

**What it does:** Handles image uploads and returns predictions

#### **Part 1: Setup**

```python
from fastapi import APIRouter, File, UploadFile, HTTPException
from ml.inference.predictor import SkinDiseasePredictor
from ml.inference.clinical_predictor import ClinicalSkinPredictor

router = APIRouter()

# Global variables for model predictors (loaded once, reused)
_dermo_predictor: SkinDiseasePredictor | None = None
_clinical_predictor: ClinicalSkinPredictor | None = None
```

**Key Concepts:**
- `APIRouter` - Groups related endpoints
- Predictors loaded once (lazy loading) - saves memory

#### **Part 2: Helper Functions**

```python
def get_dermo_predictor() -> SkinDiseasePredictor:
    """Load dermoscopic model (only once)"""
    global _dermo_predictor
    if _dermo_predictor is None:
        _dermo_predictor = SkinDiseasePredictor()
    return _dermo_predictor

def get_clinical_predictor() -> ClinicalSkinPredictor:
    """Load clinical model (only once)"""
    global _clinical_predictor
    if _clinical_predictor is None:
        _clinical_predictor = ClinicalSkinPredictor()
    return _clinical_predictor
```

**Why Lazy Loading?**
- Models are large (~100MB+)
- Only load when needed
- Load once, reuse many times

#### **Part 3: Main Endpoint**

```python
@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict skin disease from uploaded image
    
    Flow:
    1. Validate file is an image
    2. Load image
    3. Run both models
    4. Compare confidences
    5. Return best prediction
    """
```

**Step-by-Step Breakdown:**

**Step 1: Validate File**
```python
if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File must be an image")
```

**Step 2: Load Image**
```python
contents = await file.read()  # Read file bytes
image = Image.open(io.BytesIO(contents))  # Convert to PIL Image
if image.mode != "RGB":
    image = image.convert("RGB")  # Ensure RGB format
```

**Step 3: Run Both Models**
```python
dermo_pred = None
clinical_pred = None
warning = None

try:
    dermo_pred = get_dermo_predictor().predict(image)
except Exception as e:
    warning = f"Dermoscopy model error: {str(e)}"

try:
    clinical_pred = get_clinical_predictor().predict(image)
except Exception as e:
    warning += f" | Clinical model error: {str(e)}"
```

**Key Concepts:**
- `try/except` - Handle errors gracefully
- Both models run independently
- If one fails, other can still work

**Step 4: Compare Confidences**
```python
if dermo_pred is not None and clinical_pred is not None:
    # Both succeeded - compare confidences
    if dermo_pred["confidence"] >= clinical_pred["confidence"]:
        primary = dermo_pred
        secondary = clinical_pred
        primary_model_type = "dermoscopic"
    else:
        primary = clinical_pred
        secondary = dermo_pred
        primary_model_type = "clinical"
elif dermo_pred is not None:
    # Only dermoscopic model worked
    primary = dermo_pred
    primary_model_type = "dermoscopic"
else:
    # Only clinical model worked
    primary = clinical_pred
    primary_model_type = "clinical"
```

**Step 5: Return Response**
```python
return PredictionResponse(
    predicted_class=primary["class"],
    confidence=primary["confidence"],
    probabilities=primary["probabilities"],
    all_classes=primary["all_classes"],
    warning=warning,
    primary_model_type=primary_model_type,
    secondary_prediction=secondary_payload,
)
```

---

### **7. `ml/inference/predictor.py` - Dermoscopic Model**

**What it does:** Loads and uses the HAM10000 dermoscopic model

```python
class SkinDiseasePredictor:
    """Predicts from dermoscopic images (7 classes)"""
    
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
        self.model_path = model_path or settings.MODEL_PATH
        self.model = None
        self._load_model()  # Load on initialization
    
    def _load_model(self):
        """Load the .h5 model file"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = None  # Model not found
    
    def predict(self, image: Image.Image) -> Dict:
        """Make prediction on image"""
        # 1. Preprocess image
        processed = preprocess_image(image, (224, 224))
        
        # 2. Run model
        predictions = self.model.predict(processed, verbose=0)
        probabilities = predictions[0]  # Remove batch dimension
        
        # 3. Find highest probability
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])
        
        # 4. Build result
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": {self.CLASS_NAMES[i]: float(p) 
                            for i, p in enumerate(probabilities)},
            "all_classes": self.CLASS_NAMES
        }
```

**Key Concepts:**
- `keras.models.load_model()` - Loads saved model
- `model.predict()` - Runs inference
- `np.argmax()` - Finds index of maximum value
- Returns probabilities for all classes

---

### **8. `ml/inference/clinical_predictor.py` - Clinical Model**

**What it does:** Same as above, but for clinical model (23 classes)

**Differences:**
- Different model file: `./models/clinical_skin_model.h5`
- Different classes: 23 clinical conditions
- Same prediction logic

---

## 🔄 How It All Works Together

### **Complete Flow:**

```
1. User uploads image (Frontend)
   ↓
2. POST /api/v1/predict (FastAPI)
   ↓
3. prediction.py receives image
   ↓
4. Loads image with PIL
   ↓
5. Calls get_dermo_predictor().predict(image)
   ├─ predictor.py loads model
   ├─ preprocess_image() prepares image
   ├─ model.predict() runs inference
   └─ Returns prediction dict
   ↓
6. Calls get_clinical_predictor().predict(image)
   ├─ Same process, different model
   └─ Returns prediction dict
   ↓
7. Compares confidences
   ↓
8. Returns PredictionResponse (JSON)
   ↓
9. Frontend displays result
```

---

## 🏋️ Hands-On Exercises

### **Exercise 1: Understand the Flow**
1. Start the server: `python run.py`
2. Open: `http://localhost:8000/docs`
3. Try the `/api/v1/predict` endpoint
4. Watch the console logs

### **Exercise 2: Add a New Endpoint**
Add to `app/api/prediction.py`:

```python
@router.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {"message": "Backend is working!"}
```

Test: `http://localhost:8000/api/v1/test`

### **Exercise 3: Modify Response**
In `prediction.py`, add a timestamp to the response:

```python
from datetime import datetime

# In predict_image function, before return:
timestamp = datetime.now().isoformat()

return PredictionResponse(
    # ... existing fields ...
    timestamp=timestamp  # Add this
)
```

### **Exercise 4: Add Logging**
Add logging to see what's happening:

```python
import logging
logger = logging.getLogger(__name__)

# In predict_image:
logger.info(f"Received image: {file.filename}")
logger.info(f"Dermoscopic prediction: {dermo_pred['class']}")
logger.info(f"Clinical prediction: {clinical_pred['class']}")
```

### **Exercise 5: Error Handling**
Add more specific error handling:

```python
try:
    dermo_pred = get_dermo_predictor().predict(image)
except ValueError as e:
    logger.error(f"Model not loaded: {e}")
    warning = "Dermoscopic model not available"
except Exception as e:
    logger.error(f"Prediction error: {e}")
    warning = f"Prediction failed: {str(e)}"
```

---

## 📖 Resources for Learning

### **Python Basics**
- [Python.org Tutorial](https://docs.python.org/3/tutorial/)
- [Real Python](https://realpython.com/)

### **FastAPI**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)

### **Machine Learning**
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Guide](https://keras.io/guides/)

### **Image Processing**
- [PIL/Pillow Documentation](https://pillow.readthedocs.io/)

### **Best Practices**
- [Python Code Style (PEP 8)](https://pep8.org/)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)

---

## 🎓 Learning Checklist

- [ ] Understand what FastAPI is
- [ ] Know what an API endpoint is
- [ ] Understand request/response flow
- [ ] Know how images are processed
- [ ] Understand how models make predictions
- [ ] Know how both models are compared
- [ ] Can trace code from `run.py` to prediction
- [ ] Can add a new endpoint
- [ ] Can modify existing code

---

## 💬 Common Questions

### **Q: What is `async def`?**
**A:** Async functions can handle multiple requests at once. FastAPI uses async for better performance.

### **Q: Why two models?**
**A:** Dermoscopic model is specialized for dermoscopy images. Clinical model handles general photos. Running both gives better accuracy.

### **Q: What is `@router.post()`?**
**A:** Decorator that tells FastAPI this function handles POST requests to that URL.

### **Q: Why normalize images?**
**A:** Neural networks work better with pixel values between 0.0-1.0 instead of 0-255.

### **Q: What is lazy loading?**
**A:** Loading models only when first needed, not at startup. Saves memory and startup time.

---

## 🚀 Next Steps

1. **Read the code** - Start with `run.py`, then `main.py`, then `prediction.py`
2. **Run the code** - Start server and test endpoints
3. **Modify the code** - Try the exercises above
4. **Read documentation** - Check FastAPI docs for more features
5. **Build something** - Add your own features!

---

## 📝 Summary

**Backend Architecture:**
- **Entry Point**: `run.py` starts server
- **App Setup**: `main.py` creates FastAPI app
- **Configuration**: `config.py` stores settings
- **Data Models**: `schemas.py` defines data structures
- **API Logic**: `prediction.py` handles requests
- **Image Processing**: `image_processing.py` prepares images
- **ML Models**: `predictor.py` and `clinical_predictor.py` make predictions

**Key Flow:**
Request → Validation → Image Processing → Model Prediction → Response

**You're ready to explore!** Start with `run.py` and follow the code flow. 🎉

