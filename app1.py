import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Define the Streamlit app
st.title("Diabetes Prediction App")
st.write("Enter patient details to predict the likelihood of diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200)
bp = st.number_input("Blood Pressure", min_value=0, max_value=150)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100)
insulin = st.number_input("Insulin", min_value=0, max_value=900)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
age = st.number_input("Age", min_value=0, max_value=120)

# Prediction
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"The model predicts that the patient has diabetes with a probability of {probability:.2f}.")
    else:
        st.success(f"The model predicts that the patient does NOT have diabetes with a probability of {1 - probability:.2f}.")
