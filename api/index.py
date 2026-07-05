from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

# Enable CORS so your frontend HTML can talk to your backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Base route to check if backend is alive (Prevents Vercel 404 errors)
@app.get("/")
def read_root():
    return {"status": "Healthy", "message": "Heart Disease Predictor API is running!"}

# 2. Dynamic model path loading (Looks for 'trained_model.sav' in the same folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(CURRENT_DIR, "trained_model .sav")

try:
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
except FileNotFoundError:
    raise RuntimeError(f"Could not find model file at: {model_path}. Please check your file layout.")

# 3. Input Data Schema
class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float

# 4. Predict Route
@app.post("/api/predict")
def predict_heart_disease(data: HeartData):
    # Order matches the exact features used during training
    input_data = (
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs, 
        data.restecg, data.thalach, data.exang, data.oldpeak, data.slope, 
        data.ca, data.thal
    )
    
    # Process for the scikit-learn model
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    
    return {"prediction": int(prediction[0])}
