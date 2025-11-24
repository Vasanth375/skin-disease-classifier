# Quick Start Guide

## Step-by-Step Setup

### 1. Navigate to Project Directory
```bash
cd skin-disease-classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download HAM10000 Dataset

**Using Kaggle CLI:**
```bash
# Install kaggle if not already installed
pip install kaggle

# Download dataset (after setting up kaggle.json)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Extract to data/raw
# Unzip the downloaded file to ./data/raw/
```

### 5. Prepare Dataset
```python
# Run the data preparation script
python ml/training/prepare_data.py
```

Or manually organize images into class folders:
- `data/processed/actinic_keratosis/`
- `data/processed/basal_cell_carcinoma/`
- `data/processed/benign_keratosis/`
- `data/processed/dermatofibroma/`
- `data/processed/melanoma/`
- `data/processed/nevus/`
- `data/processed/vascular_lesion/`

### 6. Train the Model
```bash
python ml/training/train_model.py
```

This will create `models/skin_disease_model.h5`

### 7. Run the API
```bash
python run.py
```

Or:
```bash
uvicorn app.main:app --reload
```

### 8. Test the API

Open browser: http://localhost:8000/docs

Or use curl:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" -F "file=@path/to/image.jpg"
```

## Next Steps

1. Download and extract HAM10000 dataset
2. Organize images into class folders
3. Train the model
4. Start the API server
5. Test predictions!

