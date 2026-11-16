import streamlit as st
import numpy as np
import joblib

# Load the trained Linear Regression model
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app
st.title("🏥 Insurance Charges Prediction App")
st.subheader("Enter the following details:")

sex = st.selectbox("Gender", ["Male", "Female"], key="select_sex")
claim_amount = st.number_input("Claim Amount", min_value=0.0, format="%.2f", key="num_claim_amount")
past_consultations = st.number_input("Number of Past Consultations", min_value=0, key="num_past_consultations")
number_of_past_hospitalizations = st.number_input("Number of Past Hospitalizations", min_value=0, key="num_past_hospitalizations")
hospital_expenditure = st.number_input("Hospital Expenditure", min_value=0.0, format="%.2f", key="num_hospital_expenditure")
annual_salary = st.number_input("Annual Salary", min_value=0.0, format="%.2f", key="num_annual_salary")
children = st.number_input("Number of Children", min_value=0, max_value=5, key="num_children")
smoker = st.selectbox("Is the person a smoker?", ["No", "Yes"], key="select_smoker")

# Encode smoker (0 = No, 1 = Yes)
smoker = 1 if smoker == "Yes" else 0
sex = 1 if sex == "Female" else 0

# Predict button
if st.button("Predict Insurance Charges", key="btn_predict"):
    # Create input array
    input_data = np.array([[sex, smoker, children, number_of_past_hospitalizations, claim_amount, past_consultations, hospital_expenditure, annual_salary]])
    new_data_scaled = scaler.transform(input_data) # Scale the new data

    # Make prediction
    prediction1 = model.predict(new_data_scaled)[0]

    st.success(f"💰 Estimated Insurance Charges: ₹{prediction1:,.2f}")
