import streamlit as st
import pandas as pd
import pickle

with open("models/rf_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.title("🏦 Bank Loan Default Predictor")
st.write("Enter the applicant's details to predict if they will default on their loan.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=25)
    income = st.number_input("Annual Income ($)", min_value=0, max_value=1000000, value=50000)
    emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=3)
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])

with col2:
    loan_amnt = st.number_input("Loan Amount ($)", min_value=500, max_value=50000, value=10000)
    loan_intent = st.selectbox("Loan Intent", ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'])
    loan_grade = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=25.0, value=10.0)

cred_hist = st.slider("Credit History Length (years)", 0, 30, 4)
historical_default = st.selectbox("Has defaulted previously?", ['Y', 'N'])

if st.button("Predict Loan Status", type="primary"):
    loan_percent_income = loan_amnt / income if income > 0 else 0

    input_df = pd.DataFrame({
        'person_age': [age],
        'person_income': [income],
        'person_home_ownership': [home_ownership],
        'person_emp_length': [emp_length],
        'loan_intent': [loan_intent],
        'loan_grade': [loan_grade],
        'loan_amnt': [loan_amnt],
        'loan_int_rate': [int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_default_on_file': [historical_default],
        'cb_person_cred_hist_length': [cred_hist]
    })

    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])

    numerical_cols = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length'
    ]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ HIGH RISK: The model predicts a DEFAULT. (Risk score: {prob:.1%})")
    else:
        st.success(f"✅ LOW RISK: The model predicts the loan will be PAID. (Risk score: {prob:.1%})")
