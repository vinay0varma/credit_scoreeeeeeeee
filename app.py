# app.py
import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load models and scalers (make sure these files exist in same folder)
credit_model = tf.keras.models.load_model("ann_credit_score.h5")
fraud_model  = tf.keras.models.load_model("fraud_ann_model.h5")

credit_scaler = joblib.load("credit_scaler.save")
fraud_scaler  = joblib.load("fraud_scaler.save")

THRESHOLD = 0.23   # chosen fraud threshold

st.title("📊 Beneficiary Credit Scoring & Fraud Detection")
st.write("ANN-based credit score prediction + income verification (fraud) check")

# Inputs
annual_income = st.number_input("Annual Income", min_value=0.0, value=100000.0, step=1000.0)
joint_income  = st.number_input("Joint Income", min_value=0.0, value=50000.0, step=1000.0)
business_income = st.number_input("Business Income", min_value=0.0, value=0.0, step=1000.0)
property_value = st.number_input("Property Value", min_value=0.0, value=0.0, step=1000.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0, value=0.0, step=1000.0)
beneficiary_count = st.number_input("Family Members Count", min_value=1, value=1, step=1)
repayment_score = st.number_input("Repayment Score (numeric)", min_value=0.0, value=600.0, step=1.0)
id_verified = st.selectbox("ID Verified?", ["No", "Yes"])
document_validity = st.selectbox("Documents Valid?", ["Invalid", "Valid"])

# Convert categorical
id_verified_val = 1.0 if id_verified == "Yes" else 0.0
document_validity_val = 1.0 if document_validity == "Valid" else 0.0

if st.button("Calculate Credit Score & Fraud Risk"):

    # Prepare feature arrays (must match training feature order)
    credit_input = np.array([[annual_income, joint_income, business_income,
                              property_value, loan_amount, repayment_score,
                              beneficiary_count, 0.0]])  # last value placeholder if needed
    # Our ANN credit training used 8 features:
    # ['annual_income','joint_income','property_value','business_income',
    #  'loan_amount','repayment_score','beneficiary_count','age']
    # If you prefer to include age instead of placeholder, add a number input above and put it here.
    # For now set that 8th value=age placeholder 0.0 if your model expects that.

    # Build correct credit input shape: if your model expects (8,) replace placeholder with age input field above.
    # If your ann_credit was trained with the exact 8 features used earlier, ensure the order matches.

    # For fraud model features:
    fraud_input = np.array([[annual_income, joint_income, business_income,
                             property_value, loan_amount, beneficiary_count,
                             repayment_score, id_verified_val, document_validity_val]])

    # Scale
    credit_scaled = credit_scaler.transform(credit_input)
    fraud_scaled  = fraud_scaler.transform(fraud_input)

    # Predict
    credit_score_pred = credit_model.predict(credit_scaled)[0][0]
    fraud_prob = float(fraud_model.predict(fraud_scaled)[0][0])
    fraud_label = 1 if fraud_prob >= THRESHOLD else 0

    st.subheader("📈 Predicted Credit Score")
    st.success(f"{credit_score_pred:.2f}")

    st.subheader("⚠️ Fraud Detection")
    st.write(f"Fraud Probability: **{fraud_prob:.3f}**")
    if fraud_label:
        st.error(f"Fraud Detected (threshold={THRESHOLD}) — Manual verification suggested.")
    else:
        st.success("No fraud detected (below threshold).")
