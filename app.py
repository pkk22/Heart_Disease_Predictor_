import streamlit as st
import numpy as np
import pickle

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

st.set_page_config(page_title="Heart Disease Prediction App", layout="centered")
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the required clinical measurements below to predict heart status.")

# Create input form layout using columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3], help="0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure (trestbps in mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (chol in mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "True (1)" if x == 1 else "False (0)")
    restecg = st.selectbox("Resting Electrocardiographic Results (restecg)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3], help="0 = normal; 1 = fixed defect; 2 = reversable defect")

# Prediction logic
if st.button("Predict Heart Health Status"):
    # Gather input fields in the exact order expected by the model
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    
    # Change input data to numpy array and reshape
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.error("🚨 **Prediction:** The person is likely to have a defective heart.")
    else:
        st.success("✅ **Prediction:** The person is healthy.")
