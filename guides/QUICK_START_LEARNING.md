# Quick Start: Learning the Backend (5 Minutes)

## 🎯 Goal
Understand how the backend works in the simplest way possible.

---

## 📍 Start Here: The Big Picture

**What does the backend do?**
1. Receives an image from the user
2. Runs it through 2 AI models
3. Returns the disease prediction

**That's it!** Everything else is just details.

---

## 🗺️ The Journey (Read in Order)

### **Step 1: `run.py` (30 seconds)**
**What it does:** Starts the server

```python
python run.py  # This starts everything
```

**Think of it as:** Turning on a light switch

---

### **Step 2: `app/main.py` (1 minute)**
**What it does:** Creates the web application

```python
app = FastAPI()  # Creates the app
app.include_router(prediction.router)  # Adds the /predict endpoint
```

**Think of it as:** Building a house and adding rooms

---

### **Step 3: `app/api/prediction.py` (3 minutes) - THE MOST IMPORTANT**

**What happens here:**

```python
@router.post("/predict")
async def predict_image(file: UploadFile):
    # 1. Get the image
    image = Image.open(file)
    
    # 2. Ask model 1: "What do you think?"
    result1 = model1.predict(image)
    
    # 3. Ask model 2: "What do you think?"
    result2 = model2.predict(image)
    
    # 4. Compare: Which one is more confident?
    if result1.confidence > result2.confidence:
        return result1
    else:
        return result2
```

**Think of it as:** Asking two doctors for their opinion, then picking the more confident one

---

## 🔍 Key Files Explained Simply

| File | What It Does | Real-World Analogy |
|------|--------------|-------------------|
| `run.py` | Starts server | Turning on a computer |
| `main.py` | Sets up app | Building a restaurant |
| `prediction.py` | Handles requests | Waiter taking orders |
| `predictor.py` | Uses AI model | Doctor making diagnosis |
| `config.py` | Stores settings | Settings menu |
| `schemas.py` | Defines data format | Form template |

---

## 🎓 Learning Strategy

### **Day 1: Understand the Flow**
1. Read `run.py` (5 lines)
2. Read `main.py` (20 lines)
3. Read `prediction.py` (focus on the main function)

### **Day 2: Understand the Details**
1. How images are processed (`image_processing.py`)
2. How models work (`predictor.py`)
3. How data is structured (`schemas.py`)

### **Day 3: Try Modifying**
1. Add a simple endpoint
2. Change a response message
3. Add logging

---

## 💡 Key Concepts (Super Simple)

### **1. API Endpoint**
A URL where you can send requests.

**Example:**
- URL: `http://localhost:8000/api/v1/predict`
- You send: Image file
- You get: Prediction result

### **2. Request/Response**
- **Request**: What you send (image)
- **Response**: What you get back (prediction)

### **3. Model**
A trained AI that can recognize diseases in images.

### **4. Confidence**
How sure the model is (0% to 100%).

---

## 🧪 Try This Now

### **Exercise 1: See It Work**
```bash
# Terminal 1: Start server
python run.py

# Terminal 2: Test it
curl http://localhost:8000/health
```

### **Exercise 2: Read the Code**
Open `app/api/prediction.py` and read the `predict_image` function line by line.

### **Exercise 3: Add a Comment**
Add a comment explaining what each section does.

---

## 📚 When You're Ready for More

Read the full guide: `BACKEND_LEARNING_GUIDE.md`

It has:
- Detailed explanations
- Code breakdowns
- More exercises
- Learning resources

---

## ✅ Checklist

- [ ] I know what `run.py` does
- [ ] I know what `main.py` does  
- [ ] I understand the flow in `prediction.py`
- [ ] I can start the server
- [ ] I can test an endpoint

**If you checked all, you're ready to dive deeper!** 🎉

