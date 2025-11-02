import streamlit as st
import joblib
import numpy as np

# --- Load artifacts ---
model = joblib.load("../artifact/churn_model.pkl")
scaler = joblib.load("../artifact/scaler.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Telecom Customer Churn Prediction")
st.markdown("Predict if a customer is likely to **churn** based on input features.")

# --- Group 1: Personal & Account Info ---
st.header("Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    SeniorCitizen = 1 if SeniorCitizen == "Yes" else 0
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Partner = 1 if Partner == "Yes" else 0

with col2:
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    Dependents = 1 if Dependents == "Yes" else 0
    tenure = st.slider("Tenure (months)", 0, 100, 12)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

with col3:
    TotalCharges = st.number_input("Total Charges", min_value=0.0, value=800.0)
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    PhoneService = 1 if PhoneService == "Yes" else 0
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    MultipleLines = 1 if MultipleLines == "Yes" else 0

# --- Group 2: Online Services ---
st.header("Online Services")
col1, col2, col3 = st.columns(3)

with col1:
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineSecurity = 1 if OnlineSecurity == "Yes" else 0
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    OnlineBackup = 1 if OnlineBackup == "Yes" else 0

with col2:
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    DeviceProtection = 1 if DeviceProtection == "Yes" else 0
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    TechSupport = 1 if TechSupport == "Yes" else 0

with col3:
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingTV = 1 if StreamingTV == "Yes" else 0
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    StreamingMovies = 1 if StreamingMovies == "Yes" else 0

PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
PaperlessBilling = 1 if PaperlessBilling == "Yes" else 0

# --- Group 3: Contract, Internet & Payment ---
st.header("Contract & Payment")
col1, col2, col3 = st.columns(3)

with col1:
    Contract_OneYear = st.selectbox("Contract: One Year", ["No", "Yes"])
    Contract_OneYear = 1 if Contract_OneYear == "Yes" else 0
    Contract_TwoYear = st.selectbox("Contract: Two Year", ["No", "Yes"])
    Contract_TwoYear = 1 if Contract_TwoYear == "Yes" else 0

with col2:
    InternetService_Fiber = st.selectbox("Internet Service: Fiber optic", ["No", "Yes"])
    InternetService_Fiber = 1 if InternetService_Fiber == "Yes" else 0
    InternetService_No = st.selectbox("Internet Service: No", ["No", "Yes"])
    InternetService_No = 1 if InternetService_No == "Yes" else 0

with col3:
    PaymentMethod_CreditCard = st.selectbox("Credit Card (Automatic)", ["No", "Yes"])
    PaymentMethod_CreditCard = 1 if PaymentMethod_CreditCard == "Yes" else 0
    PaymentMethod_ElectronicCheck = st.selectbox("Electronic Check", ["No", "Yes"])
    PaymentMethod_ElectronicCheck = 1 if PaymentMethod_ElectronicCheck == "Yes" else 0
    PaymentMethod_MailedCheck = st.selectbox("Mailed Check", ["No", "Yes"])
    PaymentMethod_MailedCheck = 1 if PaymentMethod_MailedCheck == "Yes" else 0

# --- Prepare input ---
input_features = np.array([[gender, SeniorCitizen, Partner, Dependents,
                            tenure, PhoneService, MultipleLines, OnlineSecurity,
                            OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
                            StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
                            Contract_OneYear, Contract_TwoYear, InternetService_Fiber,
                            InternetService_No, PaymentMethod_CreditCard,
                            PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck]])

input_scaled = scaler.transform(input_features)

# --- Predict ---
if st.button("Predict Churn"):
    churn_prob = model.predict_proba(input_scaled)[:, 1][0]
    churn_class = model.predict(input_scaled)[0]

    st.subheader("Prediction Result:")
    if churn_class == 1:
        st.warning(f"⚠️ Customer is likely to churn! (Probability: {churn_prob:.2f})")
    else:
        st.success(f"✅ Customer is likely to stay. (Probability of churn: {churn_prob:.2f})")
