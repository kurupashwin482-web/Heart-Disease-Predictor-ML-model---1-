import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart_model.pkl')
scaler = joblib.load('scaler.pkl')  
expected_columns = joblib.load('expected_columns.pkl')  # Load the expected columns

st.title("Heart Disease Prediction App by Ashwin❤️")

st.markdown("Provide the following details to predict the likelihood of heart disease:")

age = st.slider("Age", 18, 100, 40)

gender = st.selectbox("Gender", ["Male", "Female"])

chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)

cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["Yes", "No"]
)

fasting_bs = 1 if fasting_bs == "Yes" else 0

resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])

max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)

exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])

oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 10.0, 1.0)

st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

if st.button("Predict"):
    raw_input = {
        'age': age,
        'Gender'+ gender: 1,
        'ChestPainType'+ chest_pain: 1,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG'+ resting_ecg: 1,
        'MaxHR': max_hr,
        'ExerciseAngina'+ exercise_angina: 1,
        'Oldpeak': oldpeak,
        'ST_Slope'+ st_slope: 1
    }
    
    input_df = pd.DataFrame([raw_input])
    
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
            
    input_df = input_df[expected_columns]
    
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    
    if prediction == 1:
        st.error("🩺The model predicts that you are likely to have heart disease. Please consult a healthcare professional for further evaluation.")
    else:
        st.success("💖The model predicts that you are unlikely to have heart disease. However, it's always good to maintain a healthy lifestyle and consult with a healthcare professional for regular check-ups.")