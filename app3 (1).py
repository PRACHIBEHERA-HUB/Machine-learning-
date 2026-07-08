import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load(
    "customer_segmentation.pkl"
)

df = pd.read_csv(
    "Test.csv"
)

st.set_page_config(
    page_title="Customer Segmentation"
)

st.title(
    "Customer Segmentation Dashboard"
)

st.sidebar.header(
    "Customer Details"
)

age = st.sidebar.slider(
    "Age",
    18,
    90,
    30
)

exp = st.sidebar.slider(
    "Experience",
    0,
    20,
    5
)

family = st.sidebar.slider(
    "Family Size",
    1,
    10,
    3
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

married = st.sidebar.selectbox(
    "Married",
    ["Yes","No"]
)

grad = st.sidebar.selectbox(
    "Graduated",
    ["Yes","No"]
)

profession = st.sidebar.selectbox(
    "Profession",
    df["Profession"].dropna().unique()
)

score = st.sidebar.selectbox(
    "Spending",
    ["Low","Average","High"]
)

var = st.sidebar.selectbox(
    "Var",
    df["Var_1"].dropna().unique()
)

if st.sidebar.button(
    "Predict Cluster"
):

    sample = pd.DataFrame(
        [[
            gender,
            married,
            age,
            grad,
            profession,
            exp,
            score,
            family,
            var
        ]],
        columns=[
            "Gender",
            "Ever_Married",
            "Age",
            "Graduated",
            "Profession",
            "Work_Experience",
            "Spending_Score",
            "Family_Size",
            "Var_1"
        ]
    )

    cluster = model.predict(
        sample
    )[0]

    st.success(
        f"Cluster: {cluster}"
    )


fig, ax = plt.subplots()

ax.scatter(
    df["Age"],
    df["Family_Size"]
)

ax.set_xlabel(
    "Age"
)

ax.set_ylabel(
    "Family Size"
)

st.pyplot(fig)