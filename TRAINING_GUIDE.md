# Complete Training and Prediction Guide

This guide will walk you through exploring your dataset, training the model, and making predictions.

## Quick Start

### Option 1: Automated Workflow (Recommended)

Run the complete workflow with one command:

```bash
python workflow.py --all
```

This will:
1. Check dependencies
2. Explore the dataset
3. Prepare the data
4. Train the model
5. Show you how to test the API

### Option 2: Step-by-Step Manual Process

#### Step 1: Check Dependencies

```bash
python workflow.py --check-deps
```

If any packages are missing, install them:
```bash
pip install -r requirements.txt
```

#### Step 2: Explore Your Dataset

Analyze the HAM10000 dataset to understand its structure:

```bash
python workflow.py --explore
```

Or directly:
```bash
python analyze_dataset.py
```

This will generate:
- `dataset_analysis_class_distribution.png` - Class distribution charts
- `dataset_analysis_demographics.png` - Demographic analysis
- `dataset_analysis_report.txt` - Summary report

#### Step 3: Prepare the Dataset

Organize images into class folders for training:

```bash
python workflow.py --prepare
```

Or directly:
```bash
python ml/training/prepare_data.py --dataset-dir ./dataset --processed-dir ./data/processed
```

This will:
- Read the metadata from `dataset/HAM10000_metadata.csv`
- Find images in `dataset/HAM10000_images_part_1/` and `dataset/HAM10000_images_part_2/`
- Organize them into class folders in `data/processed/`:
  - `actinic_keratosis/`
  - `basal_cell_carcinoma/`
  - `benign_keratosis/`
  - `dermatofibroma/`
  - `melanoma/`
  - `nevus/`
  - `vascular_lesion/`

#### Step 4: Train the Model

Train the CNN model on your prepared dataset:

```bash
python workflow.py --train --epochs 50
```

Or directly:
```bash
python ml/training/train_model.py
```

**Note:** Training can take 30 minutes to several hours depending on your hardware.

The model will be saved to: `./models/skin_disease_model.h5`

Training history plots will be saved to: `./models/training_history.png`

#### Step 5: Start the API Server

Once training is complete, start the FastAPI server:

```bash
python run.py
```

Or:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

#### Step 6: Make Predictions

##### Using the Interactive API Documentation (Easiest)

1. Open http://localhost:8000/docs in your browser
2. Find the `/api/v1/predict` endpoint
3. Click "Try it out"
4. Click "Choose File" and select a skin lesion image
5. Click "Execute"
6. View the prediction results with confidence scores

##### Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "accept: application/json" \
     -F "file=@path/to/your/image.jpg"
```

##### Using Python

```python
import requests

url = "http://localhost:8000/api/v1/predict"
files = {"file": open("path/to/image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

##### Using JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/api/v1/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Predicted class:', data.predicted_class);
  console.log('Confidence:', data.confidence);
  console.log('All probabilities:', data.probabilities);
});
```

## API Endpoints

### 1. Health Check
```bash
GET /health
```

### 2. Predict Skin Disease
```bash
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
  "all_classes": [
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "benign_keratosis",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "vascular_lesion"
  ]
}
```

### 3. Get Disease Classes
```bash
GET /api/v1/classes
```

### 4. Get Model Information
```bash
GET /api/v1/model/info
```

## Troubleshooting

### Issue: "Model not found" error
**Solution:** Make sure you've trained the model first. Check that `./models/skin_disease_model.h5` exists.

### Issue: "No images found" during data preparation
**Solution:** 
- Verify images are in `dataset/HAM10000_images_part_1/` and `dataset/HAM10000_images_part_2/`
- Check that `dataset/HAM10000_metadata.csv` exists
- Ensure image filenames match the `image_id` in the metadata

### Issue: Training is too slow
**Solutions:**
- Reduce number of epochs: `python workflow.py --train --epochs 20`
- Reduce batch size in `train_model.py`
- Use GPU if available (TensorFlow will automatically use it if installed)

### Issue: Out of memory during training
**Solutions:**
- Reduce batch size in `train_model.py` (change `batch_size=32` to `batch_size=16` or `8`)
- Reduce input image size (change `input_size=224` to `input_size=128`)

## Model Architecture

The CNN model consists of:
- 4 Convolutional blocks with MaxPooling
- Data augmentation layers (random flip, rotation, zoom)
- Dropout layers for regularization (0.5)
- Dense layers for classification
- Softmax output for 7 disease classes

## Dataset Information

- **Dataset**: HAM10000 (Human Against Machine with 10,000 training images)
- **Classes**: 7 types of skin lesions
- **Total Images**: ~10,015 images
- **Image Format**: JPEG
- **Image Sizes**: Variable (will be resized to 224x224 for training)

## Next Steps

1. **Improve Model Performance:**
   - Experiment with different architectures
   - Try transfer learning (ResNet, VGG, etc.)
   - Adjust hyperparameters

2. **Deploy to Production:**
   - Use a production ASGI server (Gunicorn with Uvicorn workers)
   - Add authentication/authorization
   - Set up proper logging
   - Use environment variables for configuration

3. **Add Features:**
   - Batch prediction endpoint
   - Prediction history
   - Model versioning
   - Confidence threshold filtering

## Support

For issues or questions, check:
- The generated analysis reports
- Training history plots
- API documentation at `/docs`


