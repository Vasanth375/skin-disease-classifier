# Application Flow: Image Upload to Prediction

This document explains the complete flow when a user uploads a skin disease image.

## 📋 Overview

The application uses a **dual-model system** that runs two CNN models in parallel and selects the prediction with the highest confidence. This ensures accurate predictions for both dermoscopic images (HAM10000 model) and clinical photos (Clinical model).

---

## 🔄 Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React)                              │
│                                                                   │
│  1. User selects image file                                      │
│  2. Preview shown                                                │
│  3. User clicks "Run prediction"                                 │
│  4. FormData created with image                                  │
│  5. POST request to /api/v1/predict                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND API (FastAPI)                              │
│                                                                   │
│  📍 app/api/prediction.py - /predict endpoint                  │
│                                                                   │
│  Step 1: File Validation                                        │
│    ├─ Check content_type is "image/*"                           │
│    └─ Reject if not image → HTTP 400                            │
│                                                                   │
│  Step 2: Image Loading                                          │
│    ├─ Read file bytes: await file.read()                        │
│    ├─ Open with PIL: Image.open(io.BytesIO(contents))          │
│    └─ Convert to RGB if needed: image.convert("RGB")            │
│                                                                   │
│  Step 3: Run BOTH Models (Parallel)                             │
│    ├─ Try: Dermoscopic Model (HAM10000)                          │
│    │   └─ get_dermo_predictor().predict(image)                  │
│    │                                                             │
│    └─ Try: Clinical Model (New Dataset 3)                      │
│        └─ get_clinical_predictor().predict(image)               │
│                                                                   │
│  Step 4: Compare Confidences                                     │
│    ├─ If both succeed:                                          │
│    │   ├─ Compare: dermo_pred["confidence"] vs                  │
│    │   │          clinical_pred["confidence"]                    │
│    │   ├─ Higher confidence → PRIMARY                           │
│    │   └─ Lower confidence → SECONDARY                           │
│    │                                                             │
│    ├─ If only one succeeds:                                      │
│    │   └─ That one becomes PRIMARY                               │
│    │                                                             │
│    └─ If both fail:                                             │
│        └─ Return HTTP 500 error                                 │
│                                                                   │
│  Step 5: Build Response                                         │
│    └─ Return PredictionResponse with:                            │
│        ├─ predicted_class (from primary model)                  │
│        ├─ confidence (from primary model)                        │
│        ├─ probabilities (all classes from primary)               │
│        ├─ primary_model_type ("dermoscopic" or "clinical")      │
│        ├─ secondary_prediction (if available)                   │
│        └─ warning (if any model failed)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         MODEL INFERENCE (TensorFlow/Keras)                      │
│                                                                   │
│  📍 ml/inference/predictor.py (Dermoscopic)                     │
│  📍 ml/inference/clinical_predictor.py (Clinical)                │
│                                                                   │
│  For EACH model:                                                │
│                                                                   │
│  Step 1: Preprocess Image                                       │
│    ├─ app/utils/image_processing.py                             │
│    ├─ Resize to 224x224 (LANCZOS)                               │
│    ├─ Convert PIL Image → numpy array                            │
│    ├─ Normalize: [0, 255] → [0.0, 1.0]                         │
│    └─ Add batch dimension: (1, 224, 224, 3)                     │
│                                                                   │
│  Step 2: Model Prediction                                       │
│    ├─ Load model: keras.models.load_model(model_path)            │
│    ├─ Run inference: model.predict(image_array, verbose=0)      │
│    └─ Get probabilities: predictions[0]                         │
│       └─ Shape: (num_classes,) → e.g., (7,) or (23,)            │
│                                                                   │
│  Step 3: Extract Results                                        │
│    ├─ Find max probability: np.argmax(probabilities)            │
│    ├─ Get class name: CLASS_NAMES[predicted_idx]                │
│    ├─ Get confidence: probabilities[predicted_idx]               │
│    └─ Build probabilities dict: {class: prob for all classes}    │
│                                                                   │
│  Step 4: Return Dictionary                                      │
│    └─ {                                                          │
│        "class": "melanoma",                                      │
│        "confidence": 0.95,                                       │
│        "probabilities": {"melanoma": 0.95, ...},                 │
│        "all_classes": ["actinic_keratosis", ...]                │
│      }                                                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              RESPONSE TO FRONTEND                               │
│                                                                   │
│  JSON Response:                                                 │
│  {                                                               │
│    "predicted_class": "melanoma",                                │
│    "confidence": 0.95,                                           │
│    "probabilities": {                                            │
│      "melanoma": 0.95,                                           │
│      "nevus": 0.03,                                              │
│      ...                                                          │
│    },                                                             │
│    "all_classes": [...],                                         │
│    "primary_model_type": "dermoscopic",                          │
│    "secondary_prediction": {                                     │
│      "model_type": "clinical",                                    │
│      "predicted_class": "Melanoma Skin Cancer Nevi and Moles",   │
│      "confidence": 0.87                                          │
│    },                                                             │
│    "warning": null                                               │
│  }                                                               │
│                                                                   │
│  Frontend displays:                                             │
│  ├─ Main prediction: "Melanoma" (95% confidence)                │
│  ├─ Model used: "Dermoscopic model"                              │
│  ├─ Top 5 probabilities bar chart                                │
│  └─ Secondary prediction (if available):                         │
│      "Clinical model: Melanoma Skin Cancer..." (87%)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Detailed Step-by-Step Flow

### **Phase 1: Frontend (User Interaction)**

1. **User selects image**
   - File input accepts: `image/png`, `image/jpeg`, `image/jpg`
   - Image preview shown immediately

2. **User clicks "Run prediction"**
   - Form submission triggers `handleSubmit()`
   - `FormData` created with image file
   - POST request to `http://localhost:8000/api/v1/predict`

---

### **Phase 2: Backend API (FastAPI)**

**File:** `app/api/prediction.py` → `@router.post("/predict")`

#### **Step 1: File Validation** (Lines 50-55)
```python
if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="File must be an image")
```

#### **Step 2: Image Loading** (Lines 57-63)
```python
contents = await file.read()              # Read bytes
image = Image.open(io.BytesIO(contents))  # Convert to PIL Image
if image.mode != "RGB":
    image = image.convert("RGB")          # Ensure RGB format
```

#### **Step 3: Run Both Models** (Lines 65-81)
```python
# Try dermoscopic model
try:
    dermo_pred = get_dermo_predictor().predict(image)
except Exception as e:
    warning = f"Dermoscopy model error: {str(e)}"

# Try clinical model
try:
    clinical_pred = get_clinical_predictor().predict(image)
except Exception as e:
    warning += f" | Clinical model error: {str(e)}"
```

**Key Point:** Both models run **independently**. If one fails, the other can still succeed.

#### **Step 4: Compare Confidences** (Lines 89-114)

**Scenario A: Both models succeed**
```python
if dermo_pred["confidence"] >= clinical_pred["confidence"]:
    primary = dermo_pred              # Higher confidence wins
    secondary = clinical_pred
    primary_model_type = "dermoscopic"
else:
    primary = clinical_pred           # Clinical model wins
    secondary = dermo_pred
    primary_model_type = "clinical"
```

**Scenario B: Only one model succeeds**
- That model becomes `primary`
- `secondary = None`
- Warning message added

**Scenario C: Both fail**
- Returns HTTP 500 error

#### **Step 5: Build Response** (Lines 126-134)
```python
return PredictionResponse(
    predicted_class=primary["class"],           # Main answer
    confidence=primary["confidence"],            # Main confidence
    probabilities=primary["probabilities"],      # All class probs
    all_classes=primary["all_classes"],         # Available classes
    primary_model_type=primary_model_type,      # Which model won
    secondary_prediction=secondary_payload,      # Other model's result
    warning=warning                             # Any errors
)
```

---

### **Phase 3: Model Inference**

**Files:**
- `ml/inference/predictor.py` → `SkinDiseasePredictor` (HAM10000)
- `ml/inference/clinical_predictor.py` → `ClinicalSkinPredictor` (New Dataset 3)

#### **Step 1: Preprocess Image** (`app/utils/image_processing.py`)

```python
def preprocess_image(image, target_size=(224, 224)):
    # Resize to 224x224
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image)  # Shape: (224, 224, 3)
    
    # Normalize pixel values: [0, 255] → [0.0, 1.0]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension for model input
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    
    return img_array
```

#### **Step 2: Model Prediction**

**Dermoscopic Model (HAM10000):**
- Loads: `./models/skin_disease_model.h5`
- Input shape: `(1, 224, 224, 3)`
- Output shape: `(1, 7)` → 7 classes
- Classes: `["actinic_keratosis", "basal_cell_carcinoma", "benign_keratosis", "dermatofibroma", "melanoma", "nevus", "vascular_lesion"]`

**Clinical Model (New Dataset 3):**
- Loads: `./models/clinical_skin_model.h5`
- Input shape: `(1, 224, 224, 3)`
- Output shape: `(1, 23)` → 23 classes
- Classes: `["Acne and Rosacea Photos", "Atopic Dermatitis Photos", ...]` (23 total)

```python
# Run inference
predictions = self.model.predict(processed_image, verbose=0)
probabilities = predictions[0]  # Remove batch dimension
# Example: [0.01, 0.02, 0.95, 0.01, 0.005, 0.003, 0.002]
```

#### **Step 3: Extract Results**

```python
# Find index of highest probability
predicted_idx = np.argmax(probabilities)  # e.g., 2

# Get class name
predicted_class = CLASS_NAMES[predicted_idx]  # e.g., "melanoma"

# Get confidence (probability)
confidence = float(probabilities[predicted_idx])  # e.g., 0.95

# Build probabilities dictionary for all classes
prob_dict = {
    CLASS_NAMES[i]: float(prob) 
    for i, prob in enumerate(probabilities)
}
# Example: {"actinic_keratosis": 0.01, "melanoma": 0.95, ...}
```

#### **Step 4: Return Dictionary**

```python
return {
    "class": "melanoma",
    "confidence": 0.95,
    "probabilities": {
        "actinic_keratosis": 0.01,
        "basal_cell_carcinoma": 0.02,
        "melanoma": 0.95,
        ...
    },
    "all_classes": ["actinic_keratosis", "basal_cell_carcinoma", ...]
}
```

---

### **Phase 4: Response to Frontend**

**JSON Response Example:**
```json
{
  "predicted_class": "melanoma",
  "confidence": 0.95,
  "probabilities": {
    "melanoma": 0.95,
    "nevus": 0.03,
    "basal_cell_carcinoma": 0.01,
    "actinic_keratosis": 0.005,
    "benign_keratosis": 0.003,
    "dermatofibroma": 0.001,
    "vascular_lesion": 0.001
  },
  "all_classes": [
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "vascular_lesion"
  ],
  "primary_model_type": "dermoscopic",
  "secondary_prediction": {
    "model_type": "clinical",
    "predicted_class": "Melanoma Skin Cancer Nevi and Moles",
    "confidence": 0.87
  },
  "warning": null
}
```

**Frontend Display:**
- **Main prediction:** "Melanoma" (95% confidence)
- **Model indicator:** "Predicted by: Dermoscopic model"
- **Top 5 probabilities:** Bar chart showing top classes
- **Secondary prediction:** "Alternative (Clinical model): Melanoma Skin Cancer..." (87%)
- **Warning message:** (if any model failed)

---

## 🎯 Key Design Decisions

### **1. Dual-Model System**
- **Why:** Different models excel at different image types
  - HAM10000 model → Dermoscopic images (specialized equipment)
  - Clinical model → Regular photos (phone cameras, clinical photos)
- **How:** Run both, compare confidences, pick the winner

### **2. Confidence-Based Selection**
- **Why:** The model with higher confidence is more likely to be correct
- **How:** Simple comparison: `if dermo_confidence >= clinical_confidence`

### **3. Graceful Degradation**
- **Why:** System should work even if one model is missing
- **How:** Try-catch blocks allow one model to fail without breaking the system

### **4. Lazy Loading**
- **Why:** Models are large (~100MB+), don't load until needed
- **How:** `get_dermo_predictor()` and `get_clinical_predictor()` load on first use

### **5. Same Preprocessing**
- **Why:** Both models expect same input format
- **How:** Shared `preprocess_image()` function ensures consistency

---

## ⚡ Performance Considerations

1. **Model Loading:** Models loaded once on first prediction, then cached
2. **Parallel Execution:** Both models run sequentially (could be parallelized with async)
3. **Image Preprocessing:** Fast operations (resize, normalize)
4. **Model Inference:** GPU acceleration if available (TensorFlow auto-detects)

---

## 🔧 Error Handling

| Scenario | Behavior |
|----------|----------|
| Invalid file type | HTTP 400 - "File must be an image" |
| Dermoscopic model fails | Use clinical model only, add warning |
| Clinical model fails | Use dermoscopic model only, add warning |
| Both models fail | HTTP 500 - "Both models failed" |
| Model file missing | Model returns `None`, prediction fails with clear error |

---

## 📊 Example Scenarios

### **Scenario 1: Dermoscopic Image**
- **Input:** High-quality dermoscopy image of a mole
- **Dermoscopic model:** 95% confidence → "melanoma"
- **Clinical model:** 60% confidence → "Melanoma Skin Cancer Nevi and Moles"
- **Result:** Dermoscopic model wins (higher confidence)
- **Response:** `primary_model_type: "dermoscopic"`

### **Scenario 2: Clinical Photo**
- **Input:** Phone camera photo of eczema
- **Dermoscopic model:** 45% confidence → "nevus"
- **Clinical model:** 92% confidence → "Eczema Photos"
- **Result:** Clinical model wins (higher confidence)
- **Response:** `primary_model_type: "clinical"`

### **Scenario 3: Only One Model Available**
- **Input:** Any image
- **Dermoscopic model:** Available → 85% confidence
- **Clinical model:** Missing (not trained yet)
- **Result:** Use dermoscopic model only
- **Response:** `warning: "Clinical model unavailable"`

---

## 🚀 Future Enhancements

1. **Parallel Model Execution:** Use `asyncio` to run both models simultaneously
2. **Model Ensembling:** Combine predictions instead of just picking one
3. **Confidence Thresholds:** Reject predictions below certain confidence
4. **Image Type Detection:** Pre-classify image type (dermoscopic vs clinical) before running models
5. **Caching:** Cache predictions for identical images

---

## 📝 Summary

**Flow:** User upload → API validation → Run both models → Compare confidences → Return best prediction

**Key Feature:** Dual-model system ensures accurate predictions for both specialized dermoscopic images and general clinical photos.

**Result:** User gets the most confident prediction from the appropriate model, with optional secondary prediction for comparison.

