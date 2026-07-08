import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    "salary_model.pkl"
)

st.title(
    "Salary Predictor"
    
)

age = st.number_input(
    "Age"
)

gender = st.selectbox(
    "Gender",
    [
        "Male",
        "Female"
    ]
)

edu = st.text_input(
    "Education Level"
)

job = st.text_input(
    "Job Title"
)

exp = st.number_input(
    "Years of Experience"
)

history = []

if st.button(
    "Predict Salary"
):

    data = pd.DataFrame({
        "Age":[age],
        "Gender":[gender],
        "Education Level":[edu],
        "Job Title":[job],
        "Years of Experience":[exp]
    })

    result = model.predict(
        data
    )[0]

    history.append(result)

    st.success(
        f"Predicted Salary = ₹{round(result)}"
    )

    st.write(
        "Prediction History"
    )

    st.write(
        history
    )