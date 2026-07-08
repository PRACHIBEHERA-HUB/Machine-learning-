import streamlit as st
import joblib
import pandas as pd
model = joblib.load(
"house_price_model.pkl"
)
st.title(
"🏠 House Price Prediction"
)
st.write(
"Enter details"
)
area = st.number_input(
"Area"
)
bed = st.number_input(
"Bedrooms",
min_value=1
)
bath = st.number_input(
"Bathrooms",
min_value=1
)
floor = st.number_input(
"Floors",
min_value=1
)
year = st.number_input(
"Year Built",
min_value=1900
)
location = st.selectbox(
"Location",
[
"Urban",
"Suburban",
"Rural"
]
)
condition = st.selectbox(
"Condition",
[
"Excellent",
"Good",
"Average"
]
)
garage = st.selectbox(
"Garage",
[
"Yes",
"No"
]
)
if st.button(
"Predict"
):
    data = pd.DataFrame({
"Area":[area],
"Bedrooms":[bed],
"Bathrooms":[bath],
"Floors":[floor],
"YearBuilt":[year],
"Location":[location],
"Condition":[condition],
"Garage":[garage]
})
    result = model.predict(
data
)[0]
    st.success(
f"Predicted Price ₹{result:,.0f}"
    )