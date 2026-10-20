import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("customer_churn_ann_model.keras")
scaler = joblib.load("scaler2.pkl")
model_columns = joblib.load("columns2.pkl")

st.title("⚡ Customer Churn Prediction App")
st.write("Enter the customer details below to predict whether they are likely to churn.")

# Collect User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=1)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", 
    "Mailed check", 
    "Bank transfer (automatic)", 
    "Credit card (automatic)"
])

phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])

monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=29.85)
total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=29.85)

# Predict Button
if st.button("Predict"):

    
    gender_mapped = 1 if gender == "Female" else 0
    partner_mapped = 1 if partner == "Yes" else 0
    dependents_mapped = 1 if dependents == "Yes" else 0
    phone_mapped = 1 if phone_service == "Yes" else 0
    multi_mapped = 1 if multiple_lines == "Yes" else 0
    sec_mapped = 1 if online_security == "Yes" else 0
    backup_mapped = 1 if online_backup == "Yes" else 0
    device_mapped = 1 if device_protection == "Yes" else 0
    tech_mapped = 1 if tech_support == "Yes" else 0
    tv_mapped = 1 if streaming_tv == "Yes" else 0
    movies_mapped = 1 if streaming_movies == "Yes" else 0
    paperless_mapped = 1 if paperless_billing == "Yes" else 0

   
    total_services = (
        phone_mapped + multi_mapped + sec_mapped + 
        backup_mapped + device_mapped + tech_mapped + 
        tv_mapped + movies_mapped
    )


    if tenure <= 12:
        tenure_group_val = 0
    elif tenure <= 24:
        tenure_group_val = 1
    elif tenure <= 48:
        tenure_group_val = 2
    else:
        tenure_group_val = 3

   
    input_data = {
        "gender": gender_mapped,
        "SeniorCitizen": senior_citizen,
        "Partner": partner_mapped,
        "Dependents": dependents_mapped,
        "tenure": tenure,
        "PhoneService": phone_mapped,
        "MultipleLines": multi_mapped,
        "InternetService": internet_service,
        "OnlineSecurity": sec_mapped,
        "OnlineBackup": backup_mapped,
        "DeviceProtection": device_mapped,
        "TechSupport": tech_mapped,
        "StreamingTV": tv_mapped,
        "StreamingMovies": movies_mapped,
        "Contract": contract,
        "PaperlessBilling": paperless_mapped,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "TotalServices": total_services,
        "Tenuregroup": tenure_group_val
    }

    
    df_input = pd.DataFrame([input_data])

   
    df_input = pd.get_dummies(
        df_input,
        columns=["Contract", "PaymentMethod", "InternetService", "Tenuregroup"],
        dtype=int
    )

    
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    # 7. Scale numerical columns using the exact saved MinMaxScaler matching cell [37]
    nums_cols = ["tenure", "MonthlyCharges", "TotalCharges", "TotalServices"]
    df_input[nums_cols] = scaler.transform(df_input[nums_cols])

    # 8. Make Prediction with the Keras model
    prediction_prob = model.predict(df_input)[0][0]

    # 9. Display Result
    if prediction_prob > 0.5:
        st.error(f"⚠️ The customer is likely to **CHURN**. (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"✅ The customer is likely to **STAY**. (Probability: {1 - prediction_prob:.2f})")