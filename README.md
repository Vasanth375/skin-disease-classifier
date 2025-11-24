# Skin Disease Classifier - Real-time Classification API

A FastAPI-based real-time skin disease classification system using CNN (Convolutional Neural Network) with Keras and TensorFlow. This project uses the HAM10000 dataset for training and prediction.

## Project Structure

```
skin-disease-classifier/
├── app/
│   ├── api/
│   │   └── prediction.py          # API endpoints for prediction
│   ├── core/
│   │   └── config.py              # Application configuration
│   ├── models/
│   │   └── schemas.py             # Pydantic schemas
│   ├── utils/
│   │   └── image_processing.py    # Image preprocessing utilities
│   └── main.py                    # FastAPI application entry point
├── ml/
│   ├── training/
│   │   ├── train_model.py         # Model training script
│   │   └── prepare_data.py        # Data preparation script
│   └── inference/
│       └── predictor.py           # Model inference class
├── data/
│   ├── raw/                       # Raw dataset files
│   └── processed/                 # Processed and organized images
├── models/                        # Saved model files
├── notebooks/                     # Jupyter notebooks for exploration
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
└── README.md                      # This file
```

## Features

- **Real-time Prediction**: Fast API endpoints for skin disease classification
- **CNN Model**: Deep learning model using Convolutional Neural Networks
- **7 Disease Classes**: 
  - Actinic Keratosis
  - Basal Cell Carcinoma
  - Benign Keratosis
  - Dermatofibroma
  - Melanoma
  - Nevus
  - Vascular Lesion
- **RESTful API**: FastAPI with automatic documentation
- **Image Preprocessing**: Automatic image enhancement and normalization

## Setup Instructions

### 1. Clone/Download the Project

```bash
cd skin-disease-classifier
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download HAM10000 Dataset

#### Option A: Using Kaggle CLI (Recommended)

1. Install Kaggle CLI: `pip install kaggle`
2. Get your API credentials from https://www.kaggle.com/settings
3. Place `kaggle.json` in `~/.kaggle/` (or `C:\Users\YourUsername\.kaggle\` on Windows)
4. Download dataset:
   ```bash
   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   ```

#### Option B: Manual Download

1. Download from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Extract the zip file to `./data/raw/`

### 5. Prepare the Dataset

```bash
python ml/training/prepare_data.py
```

Or manually organize images into class folders in `./data/processed/`:
- `actinic_keratosis/`
- `basal_cell_carcinoma/`
- `benign_keratosis/`
- `dermatofibroma/`
- `melanoma/`
- `nevus/`
- `vascular_lesion/`

### 6. Train the Model

```bash
python ml/training/train_model.py
```

This will:
- Build a CNN model
- Train on the dataset
- Save the model to `./models/skin_disease_model.h5`
- Generate training history plots

### 7. Configure Environment

Copy `.env.example` to `.env` and update if needed:

```bash
cp .env.example .env
```

### 8. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### 1. Health Check
```
GET /health
```

### 2. Predict Skin Disease
```
POST /api/v1/predict
Content-Type: multipart/form-data

Body: image file (JPEG/PNG)
```

**Response:**
```json
{
  "predicted_class": "melanoma",
  "confidence": 0.95,
  "probabilities": {
    "melanoma": 0.95,
    "nevus": 0.03,
    "basal_cell_carcinoma": 0.02,
    ...
  },
  "all_classes": [...]
}
```

### 3. Get Disease Classes
```
GET /api/v1/classes
```

### 4. Get Model Information
```
GET /api/v1/model/info
```

## Usage Examples

### Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Using JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Model Architecture

The CNN model consists of:
- 4 Convolutional blocks with MaxPooling
- Data augmentation layers
- Dropout for regularization
- Dense layers for classification
- Softmax output for 7 classes

## Development

### Project Structure Details

- **app/**: FastAPI application code
  - `main.py`: Application entry point and route registration
  - `api/`: API endpoint definitions
  - `core/`: Configuration and settings
  - `models/`: Pydantic schemas
  - `utils/`: Utility functions

- **ml/**: Machine learning code
  - `training/`: Model training scripts
  - `inference/`: Prediction and inference code

- **data/**: Dataset storage
  - `raw/`: Original dataset files
  - `processed/`: Organized and preprocessed images

## Notes

- The model needs to be trained before using the API
- Ensure you have sufficient GPU/CPU resources for training
- Image preprocessing includes normalization and optional enhancement
- The API supports JPEG and PNG image formats

## License

This project is for educational purposes. Please ensure you comply with Kaggle's dataset usage terms.

## Contributing

Feel free to submit issues and enhancement requests!

