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

# Load the saved model (assumes model is in the root directory)
model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))

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
