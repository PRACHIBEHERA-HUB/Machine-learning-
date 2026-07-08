import streamlit as st
import joblib
import numpy as np

model = joblib.load(
    "diabetes_model.pkl"
)

st.title(
    "🩺 Diabetes Prediction"
)

preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

if st.button(
    "Predict"
):

    data = np.array([[
        preg,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    result = model.predict(
        data
    )

    if result[0] == 1:
        st.error(
            "High Diabetes Risk"
        )

    else:
        st.success(
            "Low Diabetes Risk"
        )